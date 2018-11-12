use {back, binding_model, command, conv, pipeline, resource};
use registry::{HUB, Items, ItemsGuard, Registry};
use track::{BufferTracker, TextureTracker};
use {
    CommandBuffer, LifeGuard, RefCount, Stored, SubmissionIndex, WeaklyStored,
    TextureUsageFlags,
    BindGroupLayoutId, BlendStateId, BufferId, CommandBufferId, DepthStencilStateId,
    DeviceId, PipelineLayoutId, QueueId, RenderPipelineId, ShaderModuleId,
    TextureId, TextureViewId,
};

use hal::command::RawCommandBuffer;
use hal::queue::RawCommandQueue;
use hal::{self, Device as _Device};
//use rendy_memory::{allocator, Config, Heaps};

use std::{ffi, slice};
use std::collections::hash_map::{Entry, HashMap};
use std::sync::Mutex;
use std::sync::atomic::Ordering;


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

struct ActiveFrame<B: hal::Backend> {
    submission_index: SubmissionIndex,
    fence: B::Fence,
    resources: Vec<Resource<B>>,
}

struct DestroyedResources<B: hal::Backend> {
    /// Resources that are destroyed by the user but still referenced by
    /// other objects or command buffers.
    referenced: Vec<(ResourceId, RefCount)>,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by frames associated with a fence and a submission index.
    active: Vec<ActiveFrame<B>>,
    /// Resources that are neither referenced or used, just pending
    /// actual deletion.
    free: Vec<Resource<B>>,
}

unsafe impl<B: hal::Backend> Send for DestroyedResources<B> {}
unsafe impl<B: hal::Backend> Sync for DestroyedResources<B> {}

impl<B: hal::Backend> DestroyedResources<B> {
    fn add(&mut self, resource_id: ResourceId, life_guard: &LifeGuard) {
        self.referenced.push((resource_id, life_guard.ref_count.clone()));
    }

    fn triage_referenced(
        &mut self,
        buffer_guard: &mut ItemsGuard<resource::Buffer<B>>,
        texture_guard: &mut ItemsGuard<resource::Texture<B>>,
    ) {
        for i in (0 .. self.referenced.len()).rev() {
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
                match self.active
                    .iter_mut()
                    .find(|af| af.submission_index == submit_index)
                {
                    Some(af) => af.resources.push(resource),
                    None => self.free.push(resource),
                }
            }
        }
    }

    fn cleanup(&mut self, raw: &B::Device) {
        for i in (0 .. self.active.len()).rev() {
            if raw.get_fence_status(&self.active[i].fence).unwrap() {
                let af = self.active.swap_remove(i);
                self.free.extend(af.resources);
                raw.destroy_fence(af.fence);
            }
        }

        for resource in self.free.drain(..) {
            match resource {
                Resource::Buffer(buf) => {
                    raw.destroy_buffer(buf.raw);
                }
                Resource::Texture(tex) => {
                    raw.destroy_image(tex.raw);
                }
            }
        }
    }
}

pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    queue_group: hal::QueueGroup<B, hal::General>,
    //mem_allocator: Heaps<B::Memory>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    life_guard: LifeGuard,
    buffer_tracker: Mutex<BufferTracker>,
    texture_tracker: Mutex<TextureTracker>,
    mem_props: hal::MemoryProperties,
    pub(crate) render_passes: Mutex<HashMap<RenderPassKey, B::RenderPass>>,
    pub(crate) framebuffers: Mutex<HashMap<FramebufferKey, B::Framebuffer>>,
    last_submission_index: SubmissionIndex,
    destroyed: Mutex<DestroyedResources<B>>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
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

        Device {
            raw,
            //mem_allocator,
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            queue_group,
            life_guard: LifeGuard::new(),
            buffer_tracker: Mutex::new(BufferTracker::new()),
            texture_tracker: Mutex::new(TextureTracker::new()),
            mem_props,
            render_passes: Mutex::new(HashMap::new()),
            framebuffers: Mutex::new(HashMap::new()),
            last_submission_index: 0,
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
pub extern "C" fn wgpu_device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> TextureId {
    let kind = conv::map_texture_dimension_size(desc.dimension, desc.size);
    let format = conv::map_texture_format(desc.format);
    let aspects = format.surface_desc().aspects;
    let usage = conv::map_texture_usage(desc.usage, aspects);
    let device_guard = HUB.devices.lock();
    let device = &device_guard.get(device_id);
    let image_unbound = device
        .raw
        .create_image(
            kind,
            1, // TODO: mips
            format,
            hal::image::Tiling::Optimal, // TODO: linear
            usage,
            hal::image::ViewCapabilities::empty(), // TODO: format, 2d array, cube
        )
        .unwrap();
    let image_req = device.raw.get_image_requirements(&image_unbound);
    let device_type = device
        .mem_props
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| { // TODO
            image_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();
    // TODO: allocate with rendy
    let image_memory = device.raw.allocate_memory(device_type, image_req.size).unwrap();
    let bound_image = device
        .raw
        .bind_image_memory(&image_memory, 0, image_unbound)
        .unwrap();

    let full_range = hal::image::SubresourceRange {
        aspects,
        levels: 0 .. 1, //TODO: mips
        layers: 0 .. 1, //TODO
    };

    let life_guard = LifeGuard::new();
    let ref_count = life_guard.ref_count.clone();
    let id = HUB.textures
        .lock()
        .register(resource::Texture {
            raw: bound_image,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            kind,
            format: desc.format,
            full_range,
            life_guard,
        });
    let query = device.texture_tracker
        .lock()
        .unwrap()
        .query(
            &Stored { value: id, ref_count },
            TextureUsageFlags::WRITE_ALL,
        );
    assert!(query.initialized);

    id
}

#[no_mangle]
pub extern "C" fn wgpu_texture_create_texture_view(
    texture_id: TextureId,
    desc: &resource::TextureViewDescriptor,
) -> TextureViewId {
    let texture_guard = HUB.textures.lock();
    let texture = texture_guard.get(texture_id);

    let raw = HUB.devices
        .lock()
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
        .unwrap();

    HUB.texture_views
        .lock()
        .register(resource::TextureView {
            raw,
            texture_id: Stored {
                value: texture_id,
                ref_count: texture.life_guard.ref_count.clone(),
            },
            format: texture.format,
            extent: texture.kind.extent(),
            samples: texture.kind.num_samples(),
            life_guard: LifeGuard::new(),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_texture_create_default_texture_view(
    texture_id: TextureId,
) -> TextureViewId {
    let texture_guard = HUB.textures.lock();
    let texture = texture_guard.get(texture_id);

    let view_kind = match texture.kind {
        hal::image::Kind::D1(..) => hal::image::ViewKind::D1,
        hal::image::Kind::D2(..) => hal::image::ViewKind::D2, //TODO: array
        hal::image::Kind::D3(..) => hal::image::ViewKind::D3,
    };

    let raw = HUB.devices
        .lock()
        .get(texture.device_id.value)
        .raw
        .create_image_view(
            &texture.raw,
            view_kind,
            conv::map_texture_format(texture.format),
            hal::format::Swizzle::NO,
            texture.full_range.clone(),
        )
        .unwrap();

    HUB.texture_views
        .lock()
        .register(resource::TextureView {
            raw,
            texture_id: Stored {
                value: texture_id,
                ref_count: texture.life_guard.ref_count.clone(),
            },
            format: texture.format,
            extent: texture.kind.extent(),
            samples: texture.kind.num_samples(),
            life_guard: LifeGuard::new(),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_texture_destroy(
    texture_id: TextureId,
) {
    let texture_guard = HUB.textures.lock();
    let texture = texture_guard.get(texture_id);
    let device_guard = HUB.devices.lock();
    device_guard
        .get(texture.device_id.value)
        .destroyed
        .lock()
        .unwrap()
        .add(ResourceId::Texture(texture_id), &texture.life_guard);
}

#[no_mangle]
pub extern "C" fn wgpu_texture_view_destroy(
    _texture_view_id: TextureViewId,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length) };

    let descriptor_set_layout = HUB.devices
        .lock()
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
        .unwrap();

    HUB.bind_group_layouts
        .lock()
        .register(binding_model::BindGroupLayout {
            raw: descriptor_set_layout,
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    let bind_group_layouts = unsafe {
        slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length)
    };
    let bind_group_layout_guard = HUB.bind_group_layouts.lock();
    let descriptor_set_layouts = bind_group_layouts
        .iter()
        .map(|&id| &bind_group_layout_guard.get(id).raw);

    // TODO: push constants
    let pipeline_layout = HUB.devices
        .lock()
        .get(device_id)
        .raw
        .create_pipeline_layout(descriptor_set_layouts, &[])
        .unwrap();

    HUB.pipeline_layouts
        .lock()
        .register(binding_model::PipelineLayout {
            raw: pipeline_layout,
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_blend_state(
    _device_id: DeviceId,
    desc: &pipeline::BlendStateDescriptor,
) -> BlendStateId {
    HUB.blend_states
        .lock()
        .register(pipeline::BlendState {
            raw: conv::map_blend_state_descriptor(desc),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_depth_stencil_state(
    _device_id: DeviceId,
    desc: &pipeline::DepthStencilStateDescriptor,
) -> DepthStencilStateId {
    HUB.depth_stencil_states
        .lock()
        .register(pipeline::DepthStencilState {
            raw: conv::map_depth_stencil_state(desc),
        })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_shader_module(
    device_id: DeviceId,
    desc: &pipeline::ShaderModuleDescriptor,
) -> ShaderModuleId {
    let spv = unsafe {
        slice::from_raw_parts(desc.code.bytes, desc.code.length)
    };
    let shader = HUB.devices
        .lock()
        .get(device_id)
        .raw
        .create_shader_module(spv)
        .unwrap();

    HUB.shader_modules
        .lock()
        .register(ShaderModule { raw: shader })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_command_buffer(
    device_id: DeviceId,
    _desc: &command::CommandBufferDescriptor,
) -> CommandBufferId {
    let device_guard = HUB.devices.lock();
    let device = device_guard.get(device_id);

    let dev_stored = Stored {
        value: device_id,
        ref_count: device.life_guard.ref_count.clone(),
    };
    let mut cmd_buf = device.com_allocator.allocate(dev_stored, &device.raw);
    cmd_buf.raw.last_mut().unwrap().begin(
        hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
        hal::command::CommandBufferInheritanceInfo::default(),
    );
    HUB.command_buffers.lock().register(cmd_buf)
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
    let mut device_guard = HUB.devices.lock();
    let device = device_guard.get_mut(queue_id);
    let mut buffer_tracker = device.buffer_tracker.lock().unwrap();
    let mut texture_tracker = device.texture_tracker.lock().unwrap();

    let mut command_buffer_guard = HUB.command_buffers.lock();
    let command_buffer_ids = unsafe {
        slice::from_raw_parts(command_buffer_ptr, command_buffer_count)
    };

    let mut buffer_guard = HUB.buffers.lock();
    let mut texture_guard = HUB.textures.lock();
    let old_submit_index = device.life_guard.submission_index.fetch_add(1, Ordering::Relaxed);

    //TODO: if multiple command buffers are submitted, we can re-use the last
    // native command buffer of the previous chain instead of always creating
    // a temporary one, since the chains are not finished.

    // finish all the command buffers first
    for &cmb_id in command_buffer_ids {
        let comb = command_buffer_guard.get_mut(cmb_id);
        // update submission IDs
        for id in comb.buffer_tracker.used() {
            buffer_guard.get(id).life_guard.submission_index.store(old_submit_index, Ordering::Release);
        }
        for id in comb.texture_tracker.used() {
            texture_guard.get(id).life_guard.submission_index.store(old_submit_index, Ordering::Release);
        }

        // execute resource transitions
        let mut transit = device.com_allocator.extend(comb);
        transit.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
        //TODO: fix the consume
        CommandBuffer::insert_barriers(
            &mut transit,
            buffer_tracker.consume(&comb.buffer_tracker),
            texture_tracker.consume(&comb.texture_tracker),
        );
        transit.finish();
        comb.raw.insert(0, transit);
        comb.raw
            .last_mut()
            .unwrap()
            .finish();
    }

    // now prepare the GPU submission
    let fence = device.raw.create_fence(false).unwrap();
    {
        let submission = hal::queue::RawSubmission {
            cmd_buffers: command_buffer_ids
                .iter()
                .flat_map(|&cmb_id| {
                    &command_buffer_guard.get(cmb_id).raw
                }),
            wait_semaphores: &[],
            signal_semaphores: &[],
        };
        unsafe {
            device.queue_group.queues[0]
                .as_raw_mut()
                .submit_raw(submission, Some(&fence));
        }
    }

    if let Ok(mut destroyed) = device.destroyed.lock() {
        destroyed.triage_referenced(&mut buffer_guard, &mut texture_guard);
        destroyed.cleanup(&device.raw);

        destroyed.active.push(ActiveFrame {
            submission_index: old_submit_index + 1,
            fence,
            resources: Vec::new(),
        });
    }

    // finally, return the command buffers to the allocator
    for &cmb_id in command_buffer_ids {
        let cmd_buf = command_buffer_guard.take(cmb_id);
        device.com_allocator.submit(cmd_buf);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
) -> RenderPipelineId {
    // TODO
    let extent = hal::window::Extent2D {
        width: 100,
        height: 100,
    };

    let device_guard = HUB.devices.lock();
    let device = device_guard.get(device_id);
    let pipeline_layout_guard = HUB.pipeline_layouts.lock();
    let layout = &pipeline_layout_guard.get(desc.layout).raw;
    let pipeline_stages = unsafe { slice::from_raw_parts(desc.stages, desc.stages_length) };
    let shader_module_guard = HUB.shader_modules.lock();

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
        let depth_stencil_attachment = unsafe {
            desc.attachments_state.depth_stencil_attachment.as_ref()
        };
        let color_keys = color_attachments.iter().map(|at| hal::pass::Attachment {
            format: Some(conv::map_texture_format(at.format)),
            samples: at.samples as u8,
            ops: op_keep,
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::General .. hal::image::Layout::General,
        });
        let depth_stencil_key = depth_stencil_attachment.map(|at| hal::pass::Attachment {
            format: Some(conv::map_texture_format(at.format)),
            samples: at.samples as u8,
            ops: op_keep,
            stencil_ops: op_keep,
            layouts: hal::image::Layout::General .. hal::image::Layout::General,
        });
        RenderPassKey {
            attachments: color_keys.chain(depth_stencil_key).collect(),
        }
    };

    let mut render_pass_cache = device.render_passes.lock().unwrap();
    let main_pass = match render_pass_cache.entry(rp_key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let color_ids = [
                (0, hal::image::Layout::ColorAttachmentOptimal),
                (1, hal::image::Layout::ColorAttachmentOptimal),
                (2, hal::image::Layout::ColorAttachmentOptimal),
                (3, hal::image::Layout::ColorAttachmentOptimal),
            ];
            let depth_id = (desc.attachments_state.color_attachments_length, hal::image::Layout::DepthStencilAttachmentOptimal);

            let subpass = hal::pass::SubpassDesc {
                colors: &color_ids[.. desc.attachments_state.color_attachments_length],
                depth_stencil: if desc.attachments_state.depth_stencil_attachment.is_null() {
                    None
                } else {
                    Some(&depth_id)
                },
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let pass = device.raw.create_render_pass(
                &e.key().attachments,
                &[subpass],
                &[],
            ).unwrap();
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

    // TODO
    let vertex_buffers: Vec<hal::pso::VertexBufferDesc> = Vec::new();

    // TODO
    let attributes: Vec<hal::pso::AttributeDesc> = Vec::new();

    let input_assembler = hal::pso::InputAssemblerDesc {
        primitive: conv::map_primitive_topology(desc.primitive_topology),
        primitive_restart: hal::pso::PrimitiveRestart::Disabled, // TODO
    };

    let blend_state_guard = HUB.blend_states.lock();
    let blend_states = unsafe { slice::from_raw_parts(desc.blend_states, desc.blend_states_length) }
        .iter()
        .map(|id| blend_state_guard.get(id.clone()).raw)
        .collect();

    let blender = hal::pso::BlendDesc {
        logic_op: None, // TODO
        targets: blend_states,
    };

    let depth_stencil_state_guard = HUB.depth_stencil_states.lock();
    let depth_stencil = depth_stencil_state_guard.get(desc.depth_stencil_state).raw;

    // TODO
    let multisampling: Option<hal::pso::Multisampling> = None;

    // TODO
    let baked_states = hal::pso::BakedStates {
        viewport: Some(hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as i16,
                h: extent.height as i16,
            },
            depth: (0.0..1.0),
        }),
        scissor: Some(hal::pso::Rect {
            x: 0,
            y: 0,
            w: extent.width as i16,
            h: extent.height as i16,
        }),
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
    let pipeline = device.raw
        .create_graphics_pipeline(&pipeline_desc, None)
        .unwrap();

    HUB.render_pipelines
        .lock()
        .register(pipeline::RenderPipeline { raw: pipeline })
}
