use {back, binding_model, command, conv, pipeline, resource};
use registry::{HUB, Items, Registry};
use track::{BufferTracker, TextureTracker, TrackPermit};
use {
    AttachmentStateId, BindGroupLayoutId, BlendStateId, CommandBufferId, DepthStencilStateId,
    DeviceId, PipelineLayoutId, QueueId, RenderPipelineId, ShaderModuleId, TextureId,
};

use hal::command::RawCommandBuffer;
use hal::queue::RawCommandQueue;
use hal::{self, Device as _Device};
use rendy_memory::{allocator, Config, Heaps};

use std::{ffi, slice};
use std::sync::Mutex;


pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    queue_group: hal::QueueGroup<B, hal::General>,
    mem_allocator: Heaps<B::Memory>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    buffer_tracker: Mutex<BufferTracker>,
    texture_tracker: Mutex<TextureTracker>,
    mem_props: hal::MemoryProperties,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        queue_group: hal::QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        // TODO: These values are just taken from rendy's test
        // Need to set reasonable values per memory type instead
        let arena = Some(allocator::ArenaConfig {
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
        };

        Device {
            raw,
            mem_allocator,
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            queue_group,
            buffer_tracker: Mutex::new(BufferTracker::new()),
            texture_tracker: Mutex::new(TextureTracker::new()),
            mem_props,
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
    let usage = conv::map_texture_usage(desc.usage, format);
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

    let id = HUB.textures
        .lock()
        .register(resource::Texture {
            raw: bound_image,
        });
    device.texture_tracker
        .lock()
        .unwrap()
        .track(id, resource::TextureUsageFlags::empty(), TrackPermit::empty())
        .expect("Resource somehow is already registered");

    id
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
        );

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
        .create_pipeline_layout(descriptor_set_layouts, &[]);

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

    let mut cmd_buf = device.com_allocator.allocate(device_id, &device.raw);
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
    let mut command_buffer_guard = HUB.command_buffers.lock();
    let command_buffer_ids = unsafe {
        slice::from_raw_parts(command_buffer_ptr, command_buffer_count)
    };

    // finish all the command buffers first
    for &cmb_id in command_buffer_ids {
        command_buffer_guard
            .get_mut(cmb_id)
            .raw
            .last_mut()
            .unwrap()
            .finish();
    }

    // now prepare the GPU submission
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
                .submit_raw(submission, None);
        }
    }

    // finally, return the command buffers to the allocator
    for &cmb_id in command_buffer_ids {
        let cmd_buf = command_buffer_guard.take(cmb_id);
        device.com_allocator.submit(cmd_buf);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_attachment_state(
    device_id: DeviceId,
    desc: &pipeline::AttachmentStateDescriptor,
) -> AttachmentStateId {
    let device_guard = HUB.devices.lock();
    let device = &device_guard.get(device_id).raw;

    let color_formats = unsafe {
        slice::from_raw_parts(desc.formats, desc.formats_length)
    };
    let color_formats: Vec<_> = color_formats
        .iter()
        .cloned()
        .map(conv::map_texture_format)
        .collect();
    let depth_stencil_format = None;

    let base_pass = {
        let attachments = color_formats.iter().map(|cf| hal::pass::Attachment {
            format: Some(*cf),
            samples: 1,
            ops: hal::pass::AttachmentOps::DONT_CARE,
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::General .. hal::image::Layout::General,
        });

        let subpass = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        device.create_render_pass(
            attachments,
            &[subpass],
            &[],
        )
    };

    let at_state = pipeline::AttachmentState {
        base_pass,
        color_formats,
        depth_stencil_format,
    };

    HUB.attachment_states
        .lock()
        .register(at_state)
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
    let device = &device_guard.get(device_id).raw;
    let pipeline_layout_guard = HUB.pipeline_layouts.lock();
    let layout = &pipeline_layout_guard.get(desc.layout).raw;
    let pipeline_stages = unsafe { slice::from_raw_parts(desc.stages, desc.stages_length) };
    let shader_module_guard = HUB.shader_modules.lock();
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
    let blend_state = unsafe { slice::from_raw_parts(desc.blend_state, desc.blend_state_length) }
        .iter()
        .map(|id| blend_state_guard.get(id.clone()).raw)
        .collect();

    let blender = hal::pso::BlendDesc {
        logic_op: None, // TODO
        targets: blend_state,
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

    let attachment_state_guard = HUB.attachment_states.lock();
    let attachment_state = attachment_state_guard.get(desc.attachment_state);

    // TODO
    let subpass = hal::pass::Subpass {
        index: 0,
        main_pass: &attachment_state.base_pass,
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
    let pipeline = device
        .create_graphics_pipeline(&pipeline_desc, None)
        .unwrap();

    HUB.render_pipelines
        .lock()
        .register(pipeline::RenderPipeline { raw: pipeline })
}
