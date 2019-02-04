use crate::{back, binding_model, command, conv, pipeline, resource, swap_chain};
use crate::registry::{HUB, Items};
use crate::track::{BufferTracker, TextureTracker, TrackPermit};
use crate::{
    LifeGuard, RefCount, Stored, SubmissionIndex, WeaklyStored,
    BindGroupLayoutId, BindGroupId,
    BlendStateId, BufferId, CommandBufferId, DepthStencilStateId,
    AdapterId, DeviceId, PipelineLayoutId, QueueId, RenderPipelineId, ShaderModuleId,
    SamplerId, TextureId, TextureViewId,
    SurfaceId, SwapChainId,
};

use hal::command::RawCommandBuffer;
use hal::queue::RawCommandQueue;
use hal::{self,
    DescriptorPool as _DescriptorPool,
    Device as _Device,
    Surface as _Surface,
};
use log::trace;
//use rendy_memory::{allocator, Config, Heaps};
use parking_lot::{Mutex};

use std::{ffi, iter, slice};
use std::collections::hash_map::{Entry, HashMap};
use std::sync::atomic::Ordering;


pub fn all_buffer_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::DRAW_INDIRECT |
    Ps::VERTEX_INPUT |
    Ps::VERTEX_SHADER |
    Ps::FRAGMENT_SHADER |
    Ps::COMPUTE_SHADER |
    Ps::TRANSFER |
    Ps::HOST
}
pub fn all_image_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::EARLY_FRAGMENT_TESTS |
    Ps::LATE_FRAGMENT_TESTS |
    Ps::COLOR_ATTACHMENT_OUTPUT |
    Ps::VERTEX_SHADER |
    Ps::FRAGMENT_SHADER |
    Ps::COMPUTE_SHADER |
    Ps::TRANSFER
}

#[derive(Hash, PartialEq)]
pub(crate) struct RenderPassKey {
    pub attachments: Vec<hal::pass::Attachment>,
}
impl Eq for RenderPassKey {}

#[derive(Hash, PartialEq)]
pub(crate) struct FramebufferKey {
    pub attachments: Vec<WeaklyStored<TextureViewId>>,
}
impl Eq for FramebufferKey {}

enum ResourceId {
    Buffer(BufferId),
    Texture(TextureId),
}

enum Resource<B: hal::Backend> {
    Buffer(resource::Buffer<B>),
    Texture(resource::Texture<B>),
}

struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    resources: Vec<Resource<B>>,
}

struct DestroyedResources<B: hal::Backend> {
    /// Resources that are destroyed by the user but still referenced by
    /// other objects or command buffers.
    referenced: Vec<(ResourceId, RefCount)>,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by submissions associated with a fence and a submission index.
    active: Vec<ActiveSubmission<B>>,
    /// Resources that are neither referenced or used, just pending
    /// actual deletion.
    free: Vec<Resource<B>>,
}

unsafe impl<B: hal::Backend> Send for DestroyedResources<B> {}
unsafe impl<B: hal::Backend> Sync for DestroyedResources<B> {}

impl<B: hal::Backend> DestroyedResources<B> {
    fn add(&mut self, resource_id: ResourceId, life_guard: &LifeGuard) {
        self.referenced
            .push((resource_id, life_guard.ref_count.clone()));
    }

    fn triage_referenced<Gb, Gt>(&mut self, buffer_guard: &mut Gb, texture_guard: &mut Gt)
    where
        Gb: Items<resource::Buffer<B>>,
        Gt: Items<resource::Texture<B>>,
    {
        for i in (0..self.referenced.len()).rev() {
            // one in resource itself, and one here in this list
            let num_refs = self.referenced[i].1.load();
            if num_refs <= 2 {
                assert_eq!(num_refs, 2);
                let resource_id = self.referenced.swap_remove(i).0;
                let (submit_index, resource) = match resource_id {
                    ResourceId::Buffer(id) => {
                        let buf = buffer_guard.take(id);
                        let si = buf.life_guard.submission_index.load(Ordering::Acquire);
                        (si, Resource::Buffer(buf))
                    }
                    ResourceId::Texture(id) => {
                        let tex = texture_guard.take(id);
                        let si = tex.life_guard.submission_index.load(Ordering::Acquire);
                        (si, Resource::Texture(tex))
                    }
                };
                match self
                    .active
                    .iter_mut()
                    .find(|a| a.index == submit_index)
                {
                    Some(a) => a.resources.push(resource),
                    None => self.free.push(resource),
                }
            }
        }
    }

    /// Returns the last submission index that is done.
    fn cleanup(&mut self, raw: &B::Device) -> SubmissionIndex {
        let mut last_done = 0;

        for i in (0..self.active.len()).rev() {
            if unsafe {
                raw.get_fence_status(&self.active[i].fence).unwrap()
            } {
                let a = self.active.swap_remove(i);
                last_done = last_done.max(a.index);
                self.free.extend(a.resources);
                unsafe {
                    raw.destroy_fence(a.fence);
                }
            }
        }

        for resource in self.free.drain(..) {
            match resource {
                Resource::Buffer(buf) => {
                    unsafe { raw.destroy_buffer(buf.raw) };
                }
                Resource::Texture(tex) => {
                    unsafe { raw.destroy_image(tex.raw) };
                }
            }
        }

        last_done
    }
}


pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    adapter_id: WeaklyStored<AdapterId>,
    pub(crate) queue_group: hal::QueueGroup<B, hal::General>,
    //mem_allocator: Heaps<B::Memory>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    life_guard: LifeGuard,
    buffer_tracker: Mutex<BufferTracker>,
    pub(crate) texture_tracker: Mutex<TextureTracker>,
    mem_props: hal::MemoryProperties,
    pub(crate) render_passes: Mutex<HashMap<RenderPassKey, B::RenderPass>>,
    pub(crate) framebuffers: Mutex<HashMap<FramebufferKey, B::Framebuffer>>,
    desc_pool: Mutex<B::DescriptorPool>,
    destroyed: Mutex<DestroyedResources<B>>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: WeaklyStored<AdapterId>,
        queue_group: hal::QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        // TODO: These values are just taken from rendy's test
        // Need to set reasonable values per memory type instead
        /*let arena = Some(allocator::ArenaConfig {
            arena_size: 32 * 1024,
        });
        let dynamic = Some(allocator::DynamicConfig {
            blocks_per_chunk: 64,
            block_size_granularity: 256,
            max_block_size: 32 * 1024,
        });
        let config = Config { arena, dynamic };
        let mem_allocator = unsafe {
            Heaps::new(
                mem_props
                    .memory_types
                    .iter()
                    .map(|mt| (mt.properties.into(), mt.heap_index as u32, config)),
                mem_props.memory_heaps.clone(),
            )
        };*/

        //TODO: generic allocator for descriptors
        let desc_pool = Mutex::new(
            unsafe {
                raw.create_descriptor_pool(
                    100,
                    &[
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::Sampler,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::SampledImage,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::UniformBuffer,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::StorageBuffer,
                            count: 100,
                        },
                    ],
                )
            }.unwrap()
        );

        Device {
            raw,
            adapter_id,
            //mem_allocator,
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            queue_group,
            life_guard: LifeGuard::new(),
            buffer_tracker: Mutex::new(BufferTracker::new()),
            texture_tracker: Mutex::new(TextureTracker::new()),
            mem_props,
            render_passes: Mutex::new(HashMap::new()),
            framebuffers: Mutex::new(HashMap::new()),
            desc_pool,
            destroyed: Mutex::new(DestroyedResources {
                referenced: Vec::new(),
                active: Vec::new(),
                free: Vec::new(),
            }),
        }
    }
}


pub(crate) struct ShaderModule<B: hal::Backend> {
    pub raw: B::ShaderModule,
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
) -> BufferId {
    let device_guard = HUB.devices.read();
    let device = &device_guard.get(device_id);
    let (usage, _) = conv::map_buffer_usage(desc.usage);

    let mut buffer = unsafe {
        device.raw.create_buffer(desc.size as u64, usage).unwrap()
    };
    let requirements = unsafe {
        device.raw.get_buffer_requirements(&buffer)
    };
    let device_type = device
        .mem_props
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            // TODO
            requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();
    // TODO: allocate with rendy
    let memory = unsafe {
        device.raw
            .allocate_memory(device_type, requirements.size)
            .unwrap()
    };
    unsafe {
        device.raw
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .unwrap()
    };

    let life_guard = LifeGuard::new();
    let ref_count = life_guard.ref_count.clone();
    let id = HUB.buffers
        .write()
        .register(resource::Buffer {
            raw: buffer,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            life_guard,
        });
    let query = device.buffer_tracker
        .lock()
        .query(
            &Stored { value: id, ref_count },
            resource::BufferUsageFlags::empty(),
        );
    assert!(query.initialized);

    id
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_destroy(buffer_id: BufferId) {
    let buffer_guard = HUB.buffers.read();
    let buffer = buffer_guard.get(buffer_id);
    HUB.devices
        .read()
        .get(buffer.device_id.value)
        .destroyed
        .lock()
        .add(ResourceId::Buffer(buffer_id), &buffer.life_guard);
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> TextureId {
    let kind = conv::map_texture_dimension_size(desc.dimension, desc.size);
    let format = conv::map_texture_format(desc.format);
    let aspects = format.surface_desc().aspects;
    let usage = conv::map_texture_usage(desc.usage, aspects);
    let device_guard = HUB.devices.read();
    let device = &device_guard.get(device_id);

    let mut image = unsafe {
        device.raw.create_image(
            kind,
            1, // TODO: mips
            format,
            hal::image::Tiling::Optimal, // TODO: linear
            usage,
            hal::image::ViewCapabilities::empty(), // TODO: format, 2d array, cube
        )
    }
    .unwrap();
    let requirements = unsafe {
        device.raw.get_image_requirements(&image)
    };
    let device_type = device
        .mem_props
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            // TODO
            requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();
    // TODO: allocate with rendy
    let memory = unsafe {
        device.raw
            .allocate_memory(device_type, requirements.size)
            .unwrap()
    };
    unsafe {
        device.raw
            .bind_image_memory(&memory, 0, &mut image)
            .unwrap()
    };

    let full_range = hal::image::SubresourceRange {
        aspects,
        levels: 0..1, //TODO: mips
        layers: 0..1, //TODO
    };

    let life_guard = LifeGuard::new();
    let ref_count = life_guard.ref_count.clone();
    let id = HUB.textures
        .write()
        .register(resource::Texture {
            raw: image,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            kind,
            format: desc.format,
            full_range,
            swap_chain_link: None,
            life_guard,
        });
    let query = device.texture_tracker
        .lock()
        .query(
            &Stored { value: id, ref_count },
            resource::TextureUsageFlags::UNINITIALIZED,
        );
    assert!(query.initialized);

    id
}

#[no_mangle]
pub extern "C" fn wgpu_texture_create_texture_view(
    texture_id: TextureId,
    desc: &resource::TextureViewDescriptor,
) -> TextureViewId {
    let texture_guard = HUB.textures.read();
    let texture = texture_guard.get(texture_id);

    let raw = unsafe {
        HUB.devices
            .read()
            .get(texture.device_id.value)
            .raw
            .create_image_view(
                &texture.raw,
                conv::map_texture_view_dimension(desc.dimension),
                conv::map_texture_format(desc.format),
                hal::format::Swizzle::NO,
                hal::image::SubresourceRange {
                    aspects: conv::map_texture_aspect_flags(desc.aspect),
                    levels: desc.base_mip_level as u8 .. (desc.base_mip_level + desc.level_count) as u8,
                    layers: desc.base_array_layer as u16 .. (desc.base_array_layer + desc.array_count) as u16,
                },
            )
            .unwrap()
    };

    HUB.texture_views
        .write()
        .register(resource::TextureView {
            raw,
            texture_id: Stored {
                value: texture_id,
                ref_count: texture.life_guard.ref_count.clone(),
            },
            format: texture.format,
            extent: texture.kind.extent(),
            samples: texture.kind.num_samples(),
            is_owned_by_swap_chain: false,
            life_guard: LifeGuard::new(),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_texture_create_default_texture_view(texture_id: TextureId) -> TextureViewId {
    let texture_guard = HUB.textures.read();
    let texture = texture_guard.get(texture_id);

    let view_kind = match texture.kind {
        hal::image::Kind::D1(..) => hal::image::ViewKind::D1,
        hal::image::Kind::D2(..) => hal::image::ViewKind::D2, //TODO: array
        hal::image::Kind::D3(..) => hal::image::ViewKind::D3,
    };

    let raw = unsafe{
        HUB.devices
            .read()
            .get(texture.device_id.value)
            .raw
            .create_image_view(
                &texture.raw,
                view_kind,
                conv::map_texture_format(texture.format),
                hal::format::Swizzle::NO,
                texture.full_range.clone(),
            )
            .unwrap()
    };

    HUB.texture_views
        .write()
        .register(resource::TextureView {
            raw,
            texture_id: Stored {
                value: texture_id,
                ref_count: texture.life_guard.ref_count.clone(),
            },
            format: texture.format,
            extent: texture.kind.extent(),
            samples: texture.kind.num_samples(),
            is_owned_by_swap_chain: false,
            life_guard: LifeGuard::new(),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_texture_destroy(texture_id: TextureId) {
    let texture_guard = HUB.textures.read();
    let texture = texture_guard.get(texture_id);
    HUB.devices
        .read()
        .get(texture.device_id.value)
        .destroyed
        .lock()
        .add(ResourceId::Texture(texture_id), &texture.life_guard);
}

#[no_mangle]
pub extern "C" fn wgpu_texture_view_destroy(_texture_view_id: TextureViewId) {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_sampler(
    device_id: DeviceId, desc: &resource::SamplerDescriptor
) -> SamplerId {
    let device_guard = HUB.devices.read();
    let device = &device_guard.get(device_id);

    let info = hal::image::SamplerInfo {
        min_filter: conv::map_filter(desc.min_filter),
        mag_filter: conv::map_filter(desc.mag_filter),
        mip_filter: conv::map_filter(desc.mipmap_filter),
        wrap_mode: (
            conv::map_wrap(desc.r_address_mode),
            conv::map_wrap(desc.s_address_mode),
            conv::map_wrap(desc.t_address_mode),
        ),
        lod_bias: 0.0.into(),
        lod_range: desc.lod_min_clamp.into() .. desc.lod_max_clamp.into(),
        comparison: if desc.compare_function == resource::CompareFunction::Always {
            None
        } else {
            Some(conv::map_compare_function(desc.compare_function))
        },
        border: hal::image::PackedColor(match desc.border_color {
            resource::BorderColor::TransparentBlack => 0x00000000,
            resource::BorderColor::OpaqueBlack => 0x000000FF,
            resource::BorderColor::OpaqueWhite => 0xFFFFFFFF,
        }),
        anisotropic: hal::image::Anisotropic::Off, //TODO
    };
    let raw = unsafe {
        device.raw
            .create_sampler(info)
            .unwrap()
    };

    HUB.samplers
        .write()
        .register(resource::Sampler {
            raw
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length) };

    let descriptor_set_layout = unsafe {
        HUB.devices
            .read()
            .get(device_id)
            .raw
            .create_descriptor_set_layout(
                bindings.iter().map(|binding| {
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: binding.binding,
                        ty: conv::map_binding_type(binding.ty),
                        count: bindings.len(),
                        stage_flags: conv::map_shader_stage_flags(binding.visibility),
                        immutable_samplers: false, // TODO
                    }
                }),
                &[],
            )
    }
    .unwrap();

    HUB.bind_group_layouts
        .write()
        .register(binding_model::BindGroupLayout {
            raw: descriptor_set_layout,
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    let bind_group_layout_ids = unsafe {
        slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length)
    };
    let bind_group_layout_guard = HUB.bind_group_layouts.read();
    let descriptor_set_layouts = bind_group_layout_ids
        .iter()
        .map(|&id| &bind_group_layout_guard.get(id).raw);

    // TODO: push constants
    let pipeline_layout = unsafe {
        HUB.devices
            .read()
            .get(device_id)
            .raw
            .create_pipeline_layout(descriptor_set_layouts, &[])
    }
    .unwrap();

    HUB.pipeline_layouts
        .write()
        .register(binding_model::PipelineLayout {
            raw: pipeline_layout,
            bind_group_layout_ids: bind_group_layout_ids
                .iter()
                .cloned()
                .map(WeaklyStored)
                .collect(),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group(
    device_id: DeviceId,
    desc: &binding_model::BindGroupDescriptor,
) -> BindGroupId {
    let device_guard = HUB.devices.read();
    let device = device_guard.get(device_id);
    let bind_group_layout_guard = HUB.bind_group_layouts.read();
    let bind_group_layout = bind_group_layout_guard.get(desc.layout);
    let bindings = unsafe {
        slice::from_raw_parts(desc.bindings, desc.bindings_length as usize)
    };

    let mut desc_pool = device.desc_pool.lock();
    let desc_set = unsafe {
        desc_pool
            .allocate_set(&bind_group_layout.raw)
            .unwrap()
    };

    let buffer_guard = HUB.buffers.read();
    let sampler_guard = HUB.samplers.read();
    let texture_view_guard = HUB.texture_views.read();

    //TODO: group writes into contiguous sections
    let mut writes = Vec::new();
    let mut used_buffers = BufferTracker::new();
    let mut used_textures = TextureTracker::new();
    for b in bindings {
        let descriptor = match b.resource {
            binding_model::BindingResource::Buffer(ref bb) => {
                let (buffer, _) = used_buffers
                    .get_with_usage(
                        &*buffer_guard,
                        bb.buffer,
                        resource::BufferUsageFlags::UNIFORM,
                        TrackPermit::EXTEND,
                    )
                    .unwrap();
                let range = Some(bb.offset as u64) .. Some((bb.offset + bb.size) as u64);
                hal::pso::Descriptor::Buffer(&buffer.raw, range)
            }
            binding_model::BindingResource::Sampler(id) => {
                let sampler = sampler_guard.get(id);
                hal::pso::Descriptor::Sampler(&sampler.raw)
            }
            binding_model::BindingResource::TextureView(id) => {
                let view = texture_view_guard.get(id);
                used_textures
                    .transit(
                        view.texture_id.value,
                        &view.texture_id.ref_count,
                        resource::TextureUsageFlags::SAMPLED,
                        TrackPermit::EXTEND,
                    )
                    .unwrap();
                hal::pso::Descriptor::Image(&view.raw, hal::image::Layout::ShaderReadOnlyOptimal)
            }
        };
        let write = hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: b.binding,
            array_offset: 0, //TODO
            descriptors: iter::once(descriptor),
        };
        writes.push(write);
    }

    unsafe {
        device.raw.write_descriptor_sets(writes);
    }

    HUB.bind_groups
        .write()
        .register(binding_model::BindGroup {
            raw: desc_set,
            layout_id: WeaklyStored(desc.layout),
            life_guard: LifeGuard::new(),
            used_buffers,
            used_textures,
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_blend_state(
    _device_id: DeviceId,
    desc: &pipeline::BlendStateDescriptor,
) -> BlendStateId {
    HUB.blend_states.write().register(pipeline::BlendState {
        raw: conv::map_blend_state_descriptor(desc),
    })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_depth_stencil_state(
    _device_id: DeviceId,
    desc: &pipeline::DepthStencilStateDescriptor,
) -> DepthStencilStateId {
    HUB.depth_stencil_states
        .write()
        .register(pipeline::DepthStencilState {
            raw: conv::map_depth_stencil_state(desc),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_shader_module(
    device_id: DeviceId,
    desc: &pipeline::ShaderModuleDescriptor,
) -> ShaderModuleId {
    let spv = unsafe { slice::from_raw_parts(desc.code.bytes, desc.code.length) };
    let shader = unsafe {
        HUB.devices
            .read()
            .get(device_id)
            .raw
            .create_shader_module(spv)
    }
    .unwrap();

    HUB.shader_modules
        .write()
        .register(ShaderModule { raw: shader })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_command_buffer(
    device_id: DeviceId,
    _desc: &command::CommandBufferDescriptor,
) -> CommandBufferId {
    let device_guard = HUB.devices.read();
    let device = device_guard.get(device_id);

    let dev_stored = Stored {
        value: device_id,
        ref_count: device.life_guard.ref_count.clone(),
    };
    let mut cmd_buf = device.com_allocator.allocate(dev_stored, &device.raw);
    unsafe {
        cmd_buf.raw.last_mut().unwrap().begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
    }
    HUB.command_buffers.write().register(cmd_buf)
}

#[no_mangle]
pub extern "C" fn wgpu_device_get_queue(device_id: DeviceId) -> QueueId {
    device_id
}

#[no_mangle]
pub extern "C" fn wgpu_queue_submit(
    queue_id: QueueId,
    command_buffer_ptr: *const CommandBufferId,
    command_buffer_count: usize,
) {
    let mut device_guard = HUB.devices.write();
    let device = device_guard.get_mut(queue_id);
    let mut buffer_tracker = device.buffer_tracker.lock();
    let mut texture_tracker = device.texture_tracker.lock();

    let mut command_buffer_guard = HUB.command_buffers.write();
    let command_buffer_ids =
        unsafe { slice::from_raw_parts(command_buffer_ptr, command_buffer_count) };

    let mut buffer_guard = HUB.buffers.write();
    let mut texture_guard = HUB.textures.write();
    let old_submit_index = device
        .life_guard
        .submission_index
        .fetch_add(1, Ordering::Relaxed);

    let mut swap_chain_links = Vec::new();

    //TODO: if multiple command buffers are submitted, we can re-use the last
    // native command buffer of the previous chain instead of always creating
    // a temporary one, since the chains are not finished.

    // finish all the command buffers first
    for &cmb_id in command_buffer_ids {
        let comb = command_buffer_guard.get_mut(cmb_id);
        swap_chain_links.extend(comb.swap_chain_links.drain(..));
        // update submission IDs
        comb.life_guard.submission_index
            .store(old_submit_index, Ordering::Release);
        for id in comb.buffer_tracker.used() {
            buffer_guard
                .get(id)
                .life_guard
                .submission_index
                .store(old_submit_index, Ordering::Release);
        }
        for id in comb.texture_tracker.used() {
            texture_guard
                .get(id)
                .life_guard
                .submission_index
                .store(old_submit_index, Ordering::Release);
        }

        // execute resource transitions
        let mut transit = device.com_allocator.extend(comb);
        unsafe {
            transit.begin(
                hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
                hal::command::CommandBufferInheritanceInfo::default(),
            );
        }
        //TODO: fix the consume
        command::CommandBuffer::insert_barriers(
            &mut transit,
            buffer_tracker.consume_by_replace(&comb.buffer_tracker),
            texture_tracker.consume_by_replace(&comb.texture_tracker),
            &*buffer_guard,
            &*texture_guard,
        );
        unsafe {
            transit.finish();
        }
        comb.raw.insert(0, transit);
        unsafe {
            comb.raw.last_mut().unwrap().finish();
        }
    }

    // now prepare the GPU submission
    let fence = device.raw.create_fence(false).unwrap();
    {
        let swap_chain_guard = HUB.swap_chains.read();
        let wait_semaphores = swap_chain_links
            .into_iter()
            .map(|link| {
                //TODO: check the epoch
                let sem = &swap_chain_guard
                    .get(link.swap_chain_id.0)
                    .frames[link.image_index as usize]
                    .sem_available;
                (sem, hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT)
            });
        let submission =
            hal::queue::Submission::<_, _, &[<back::Backend as hal::Backend>::Semaphore]> {
                //TODO: may `OneShot` be enough?
                command_buffers: command_buffer_ids
                    .iter()
                    .flat_map(|&cmb_id| &command_buffer_guard.get(cmb_id).raw),
                wait_semaphores,
                signal_semaphores: &[], //TODO: signal `sem_present`?
            };
        unsafe {
            device.queue_group.queues[0]
                .as_raw_mut()
                .submit(submission, Some(&fence));
        }
    }

    let last_done = {
        let mut destroyed = device.destroyed.lock();
        destroyed.triage_referenced(&mut *buffer_guard, &mut *texture_guard);
        let last_done = destroyed.cleanup(&device.raw);

        destroyed.active.push(ActiveSubmission {
            index: old_submit_index + 1,
            fence,
            resources: Vec::new(),
        });

        last_done
    };

    device.com_allocator.maintain(last_done);

    // finally, return the command buffers to the allocator
    for &cmb_id in command_buffer_ids {
        let cmd_buf = command_buffer_guard.take(cmb_id);
        device.com_allocator.after_submit(cmd_buf);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
) -> RenderPipelineId {
    let device_guard = HUB.devices.read();
    let device = device_guard.get(device_id);
    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let layout = &pipeline_layout_guard.get(desc.layout).raw;
    let pipeline_stages = unsafe { slice::from_raw_parts(desc.stages, desc.stages_length) };
    let shader_module_guard = HUB.shader_modules.read();

    let rp_key = {
        let op_keep = hal::pass::AttachmentOps {
            load: hal::pass::AttachmentLoadOp::Load,
            store: hal::pass::AttachmentStoreOp::Store,
        };
        let color_attachments = unsafe {
            slice::from_raw_parts(
                desc.attachments_state.color_attachments,
                desc.attachments_state.color_attachments_length,
            )
        };
        let depth_stencil_attachment =
            unsafe { desc.attachments_state.depth_stencil_attachment.as_ref() };
        let color_keys = color_attachments.iter().map(|at| hal::pass::Attachment {
            format: Some(conv::map_texture_format(at.format)),
            samples: at.samples as u8,
            ops: op_keep,
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::General..hal::image::Layout::General,
        });
        let depth_stencil_key = depth_stencil_attachment.map(|at| hal::pass::Attachment {
            format: Some(conv::map_texture_format(at.format)),
            samples: at.samples as u8,
            ops: op_keep,
            stencil_ops: op_keep,
            layouts: hal::image::Layout::General..hal::image::Layout::General,
        });
        RenderPassKey {
            attachments: color_keys.chain(depth_stencil_key).collect(),
        }
    };

    let mut render_pass_cache = device.render_passes.lock();
    let main_pass = match render_pass_cache.entry(rp_key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let color_ids = [
                (0, hal::image::Layout::ColorAttachmentOptimal),
                (1, hal::image::Layout::ColorAttachmentOptimal),
                (2, hal::image::Layout::ColorAttachmentOptimal),
                (3, hal::image::Layout::ColorAttachmentOptimal),
            ];
            let depth_id = (
                desc.attachments_state.color_attachments_length,
                hal::image::Layout::DepthStencilAttachmentOptimal,
            );

            let subpass = hal::pass::SubpassDesc {
                colors: &color_ids[..desc.attachments_state.color_attachments_length],
                depth_stencil: if desc.attachments_state.depth_stencil_attachment.is_null() {
                    None
                } else {
                    Some(&depth_id)
                },
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let pass = unsafe {
                device
                    .raw
                    .create_render_pass(&e.key().attachments, &[subpass], &[])
            }
            .unwrap();
            e.insert(pass)
        }
    };

    let shaders = {
        let mut vertex = None;
        let mut fragment = None;
        for pipeline_stage in pipeline_stages.iter() {
            let entry = hal::pso::EntryPoint::<back::Backend> {
                entry: unsafe { ffi::CStr::from_ptr(pipeline_stage.entry_point) }
                    .to_str()
                    .to_owned()
                    .unwrap(), // TODO
                module: &shader_module_guard.get(pipeline_stage.module).raw,
                specialization: hal::pso::Specialization {
                    // TODO
                    constants: &[],
                    data: &[],
                },
            };
            match pipeline_stage.stage {
                pipeline::ShaderStage::Vertex => {
                    vertex = Some(entry);
                }
                pipeline::ShaderStage::Fragment => {
                    fragment = Some(entry);
                }
                pipeline::ShaderStage::Compute => unimplemented!(), // TODO
            }
        }

        hal::pso::GraphicsShaderSet {
            vertex: vertex.unwrap(), // TODO
            hull: None,
            domain: None,
            geometry: None,
            fragment,
        }
    };

    // TODO
    let rasterizer = hal::pso::Rasterizer {
        depth_clamping: false,
        polygon_mode: hal::pso::PolygonMode::Fill,
        cull_face: hal::pso::Face::BACK,
        front_face: hal::pso::FrontFace::Clockwise,
        depth_bias: None,
        conservative: false,
    };

    let desc_vbs = unsafe {
        slice::from_raw_parts(desc.vertex_buffer_state.vertex_buffers, desc.vertex_buffer_state.vertex_buffers_count)
    };
    let mut vertex_buffers: Vec<hal::pso::VertexBufferDesc> = Vec::with_capacity(desc_vbs.len());
    let mut attributes: Vec<hal::pso::AttributeDesc> = Vec::new();
    for (i, vb_state) in desc_vbs.iter().enumerate() {
        if vb_state.attributes_count == 0 {
            continue
        }
        vertex_buffers.push(hal::pso::VertexBufferDesc {
            binding: i as u32,
            stride: vb_state.stride,
            rate: match vb_state.step_mode {
                pipeline::InputStepMode::Vertex => 0,
                pipeline::InputStepMode::Instance => 1,
            },
        });
        let desc_atts = unsafe {
            slice::from_raw_parts(vb_state.attributes, vb_state.attributes_count)
        };
        for attribute in desc_atts {
            attributes.push(hal::pso::AttributeDesc {
                location: attribute.attribute_index,
                binding: i as u32,
                element: hal::pso::Element {
                    format: conv::map_vertex_format(attribute.format),
                    offset: attribute.offset,
                },
            });
        }
    }

    let input_assembler = hal::pso::InputAssemblerDesc {
        primitive: conv::map_primitive_topology(desc.primitive_topology),
        primitive_restart: hal::pso::PrimitiveRestart::Disabled, // TODO
    };

    let blend_state_guard = HUB.blend_states.read();
    let blend_states =
        unsafe { slice::from_raw_parts(desc.blend_states, desc.blend_states_length) }
            .iter()
            .map(|id| blend_state_guard.get(id.clone()).raw)
            .collect();

    let blender = hal::pso::BlendDesc {
        logic_op: None, // TODO
        targets: blend_states,
    };

    let depth_stencil_state_guard = HUB.depth_stencil_states.read();
    let depth_stencil = depth_stencil_state_guard.get(desc.depth_stencil_state).raw;

    // TODO
    let multisampling: Option<hal::pso::Multisampling> = None;

    // TODO
    let baked_states = hal::pso::BakedStates {
        viewport: None,
        scissor: None,
        blend_color: None,
        depth_bounds: None,
    };

    let subpass = hal::pass::Subpass {
        index: 0,
        main_pass,
    };

    // TODO
    let flags = hal::pso::PipelineCreationFlags::empty();
    // TODO
    let parent = hal::pso::BasePipeline::None;

    let pipeline_desc = hal::pso::GraphicsPipelineDesc {
        shaders,
        rasterizer,
        vertex_buffers,
        attributes,
        input_assembler,
        blender,
        depth_stencil,
        multisampling,
        baked_states,
        layout,
        subpass,
        flags,
        parent,
    };

    // TODO: cache
    let pipeline = unsafe { device.raw.create_graphics_pipeline(&pipeline_desc, None) }.unwrap();

    HUB.render_pipelines
        .write()
        .register(pipeline::RenderPipeline {
            raw: pipeline,
            layout_id: WeaklyStored(desc.layout),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_swap_chain(
    device_id: DeviceId,
    surface_id: SurfaceId,
    desc: &swap_chain::SwapChainDescriptor,
) -> SwapChainId {
    let device_guard = HUB.devices.read();
    let device = device_guard.get(device_id);
    let mut surface_guard = HUB.surfaces.write();
    let surface = surface_guard.get_mut(surface_id);

    let (caps, formats, _present_modes, _composite_alphas) = {
        let adapter_guard = HUB.adapters.read();
        let adapter = adapter_guard.get(device.adapter_id.0);
        assert!(surface.raw.supports_queue_family(&adapter.queue_families[0]));
        surface.raw.compatibility(&adapter.physical_device)
    };
    let num_frames = caps.image_count.start; //TODO: configure?
    let frame_format = conv::map_texture_format(desc.format);
    let config = hal::SwapchainConfig::new(
        desc.width,
        desc.height,
        frame_format,
        num_frames, //TODO: configure?
    );

    let usage = conv::map_texture_usage(desc.usage, hal::format::Aspects::COLOR);
    if let Some(formats) = formats {
        assert!(formats.contains(&config.format),
            "Requested format {:?} is not in supported list: {:?}",
            config.format, formats);
    }
    //TODO: properly exclusive range
    assert!(desc.width >= caps.extents.start.width && desc.width <= caps.extents.end.width &&
        desc.height >= caps.extents.start.height && desc.height <= caps.extents.end.height,
        "Requested size {}x{} is outside of the supported range: {:?}",
        desc.width, desc.height, caps.extents);

    let (raw, backbuffer) = unsafe {
        device.raw
            .create_swapchain(
                &mut surface.raw,
                config.with_image_usage(usage),
                None,
            )
            .unwrap()
    };
    let command_pool = unsafe {
        device.raw
            .create_command_pool_typed(
                &device.queue_group,
                hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
            )
            .unwrap()
    };

    let mut swap_chain_guard = HUB.swap_chains.write();

    let swap_chain_id = swap_chain_guard
        .register(swap_chain::SwapChain {
            raw,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            frames: Vec::with_capacity(num_frames as usize),
            acquired: Vec::with_capacity(1), //TODO: get it from gfx-hal?
            sem_available: device.raw.create_semaphore().unwrap(),
            command_pool,
        });
    let swap_chain = swap_chain_guard.get_mut(swap_chain_id);

    let images = match backbuffer {
        hal::Backbuffer::Images(images) => images,
        hal::Backbuffer::Framebuffer(_) => panic!("Deprecated API detected!"),
    };

    let mut texture_guard = HUB.textures.write();
    let mut texture_view_guard = HUB.texture_views.write();

    for (i, image) in images.into_iter().enumerate() {
        let kind = hal::image::Kind::D2(desc.width, desc.height, 1, 1);
        let full_range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: 0 .. 1,
        };
        let view_raw = unsafe {
            device.raw
                .create_image_view(
                    &image,
                    hal::image::ViewKind::D2,
                    frame_format,
                    hal::format::Swizzle::NO,
                    full_range.clone(),
                )
                .unwrap()
            };
        let texture = resource::Texture {
            raw: image,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            kind,
            format: desc.format,
            full_range,
            swap_chain_link: Some(swap_chain::SwapChainLink {
                swap_chain_id: WeaklyStored(swap_chain_id), //TODO: strongly
                epoch: Mutex::new(0),
                image_index: i as hal::SwapImageIndex,
            }),
            life_guard: LifeGuard::new(),
        };
        let texture_id = Stored {
            ref_count: texture.life_guard.ref_count.clone(),
            value: texture_guard.register(texture),
        };
        device.texture_tracker
            .lock()
            .query(&texture_id, resource::TextureUsageFlags::UNINITIALIZED);

        let view = resource::TextureView {
            raw: view_raw,
            texture_id: texture_id.clone(),
            format: desc.format,
            extent: kind.extent(),
            samples: kind.num_samples(),
            is_owned_by_swap_chain: true,
            life_guard: LifeGuard::new(),
        };
        swap_chain.frames.push(swap_chain::Frame {
            texture_id,
            view_id: Stored {
                 ref_count: view.life_guard.ref_count.clone(),
                 value: texture_view_guard.register(view),
            },
            fence: device.raw.create_fence(true).unwrap(),
            sem_available: device.raw.create_semaphore().unwrap(),
            sem_present: device.raw.create_semaphore().unwrap(),
            comb: swap_chain.command_pool.acquire_command_buffer(),
        });
    }

    swap_chain_id
}



#[no_mangle]
pub extern "C" fn wgpu_buffer_set_sub_data(
    buffer_id: BufferId,
    start: u32, count: u32, data: *const u8,
) {
    let buffer_guard = HUB.buffers.read();
    let buffer = buffer_guard.get(buffer_id);
    let mut device_guard = HUB.devices.write();
    let device = device_guard.get_mut(buffer.device_id.value);

    //Note: this is just doing `update_buffer`, which is limited to 64KB

    trace!("transit {:?} to transfer dst", buffer_id);
    let barrier = device.buffer_tracker
        .lock()
        .transit(
            buffer_id,
            &buffer.life_guard.ref_count,
            resource::BufferUsageFlags::TRANSFER_DST,
            TrackPermit::REPLACE,
        )
        .unwrap()
        .into_source()
        .map(|old| hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(old) ..
                hal::buffer::State::TRANSFER_WRITE,
            target: &buffer.raw,
            families: None,
            range: None .. None, //TODO: could be partial
        });

    let mut comb = device.com_allocator.allocate(buffer.device_id.clone(), &device.raw);
    unsafe {
        let raw = comb.raw.last_mut().unwrap();
        raw.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
        raw.pipeline_barrier(
            all_buffer_stages() .. hal::pso::PipelineStage::TRANSFER,
            hal::memory::Dependencies::empty(),
            barrier,
        );
        raw.update_buffer(
            &buffer.raw,
            start as hal::buffer::Offset,
            slice::from_raw_parts(data, count as usize),
        );
        raw.finish();

        let submission = hal::queue::Submission {
            command_buffers: iter::once(&*raw),
            wait_semaphores: None,
            signal_semaphores: None,
        };
        device.queue_group.queues[0]
            .as_raw_mut()
            .submit::<_, _, <back::Backend as hal::Backend>::Semaphore, _, _>(submission, None);
    }

    device.com_allocator.after_submit(comb);
}
