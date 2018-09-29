extern crate wgpu_native as wgn;
extern crate arrayvec;

use arrayvec::ArrayVec;

use std::ffi::CString;

pub use wgn::{
    AdapterDescriptor, Color, CommandBufferDescriptor, DeviceDescriptor, Extensions, Extent3d,
    Origin3d, PowerPreference, ShaderModuleDescriptor, ShaderStage,
    BindGroupLayoutBinding, TextureFormat,
    PrimitiveTopology, BlendStateDescriptor, ColorWriteFlags, DepthStencilStateDescriptor,
};


pub struct Instance {
    id: wgn::InstanceId,
}

pub struct Adapter {
    id: wgn::AdapterId,
}

pub struct Device {
    id: wgn::DeviceId,
}

pub struct BindGroupLayout {
    id: wgn::BindGroupLayoutId,
}

pub struct ShaderModule {
    id: wgn::ShaderModuleId,
}

pub struct PipelineLayout {
    id: wgn::PipelineLayoutId,
}

pub struct BlendState {
    id: wgn::BlendStateId,
}

pub struct DepthStencilState {
    id: wgn::DepthStencilStateId,
}

pub struct AttachmentState {
    id: wgn::AttachmentStateId,
}

pub struct RenderPipeline {
    id: wgn::RenderPipelineId,
}

pub struct CommandBuffer {
    id: wgn::CommandBufferId,
}

pub struct Queue {
    id: wgn::QueueId,
}

pub struct BindGroupLayoutDescriptor<'a> {
    pub bindings: &'a [BindGroupLayoutBinding],
}

pub struct PipelineLayoutDescriptor<'a> {
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

pub struct PipelineStageDescriptor<'a> {
    pub module: &'a ShaderModule,
    pub stage: ShaderStage,
    pub entry_point: &'a str,
}

pub struct AttachmentStateDescriptor<'a> {
    pub formats: &'a [TextureFormat],
}

pub struct RenderPipelineDescriptor<'a> {
    pub layout: &'a PipelineLayout,
    pub stages: &'a [PipelineStageDescriptor<'a>],
    pub primitive_topology: PrimitiveTopology,
    pub blend_state: &'a [&'a BlendState],
    pub depth_stencil_state: &'a DepthStencilState,
    pub attachment_state: &'a AttachmentState,
}


impl Instance {
    pub fn new() -> Self {
        Instance {
            id: wgn::wgpu_create_instance(),
        }
    }

    pub fn get_adapter(&self, desc: AdapterDescriptor) -> Adapter {
        Adapter {
            id: wgn::wgpu_instance_get_adapter(self.id, desc),
        }
    }
}

impl Adapter {
    pub fn create_device(&self, desc: DeviceDescriptor) -> Device {
        Device {
            id: wgn::wgpu_adapter_create_device(self.id, desc),
        }
    }
}

impl Device {
    pub fn create_shader_module(&self, spv: &[u8]) -> ShaderModule {
        let desc = wgn::ShaderModuleDescriptor {
            code: wgn::ByteArray {
                bytes: spv.as_ptr(),
                length: spv.len(),
            },
        };
        ShaderModule {
            id: wgn::wgpu_device_create_shader_module(self.id, desc),
        }
    }

    pub fn get_queue(&self) -> Queue {
        Queue {
            id: wgn::wgpu_device_get_queue(self.id),
        }
    }

    pub fn create_command_buffer(&self, desc: CommandBufferDescriptor) -> CommandBuffer {
        CommandBuffer {
            id: wgn::wgpu_device_create_command_buffer(self.id, desc),
        }
    }

    pub fn create_bind_group_layout(&self, desc: BindGroupLayoutDescriptor) -> BindGroupLayout {
        BindGroupLayout {
            id: wgn::wgpu_device_create_bind_group_layout(self.id, wgn::BindGroupLayoutDescriptor {
                bindings: desc.bindings.as_ptr(),
                bindings_length: desc.bindings.len(),
            }),
        }
    }

    pub fn create_pipeline_layout(&self, desc: PipelineLayoutDescriptor) -> PipelineLayout {
        PipelineLayout {
            id: wgn::wgpu_device_create_pipeline_layout(self.id, wgn::PipelineLayoutDescriptor {
                bind_group_layouts: desc.bind_group_layouts.as_ptr() as *const _,
                bind_group_layouts_length: desc.bind_group_layouts.len(),
            }),
        }
    }

    pub fn create_blend_state(&self, desc: BlendStateDescriptor) -> BlendState {
        BlendState {
            id: wgn::wgpu_device_create_blend_state(self.id, desc),
        }
    }

    pub fn create_depth_stencil_state(&self, desc: DepthStencilStateDescriptor) -> DepthStencilState {
        DepthStencilState {
            id: wgn::wgpu_device_create_depth_stencil_state(self.id, desc),
        }
    }

    pub fn create_attachment_state(&self, desc: AttachmentStateDescriptor) -> AttachmentState {
        AttachmentState {
            id: wgn::wgpu_device_create_attachment_state(self.id, wgn::AttachmentStateDescriptor {
                formats: desc.formats.as_ptr(),
                formats_length: desc.formats.len(),
            }),
        }
    }

    pub fn create_render_pipeline(&self, desc: RenderPipelineDescriptor) -> RenderPipeline {
        let entry_points = desc.stages
            .iter()
            .map(|ps| CString::new(ps.entry_point).unwrap())
            .collect::<ArrayVec<[_; 2]>>();
        let stages = desc.stages
            .iter()
            .zip(&entry_points)
            .map(|(ps, ep_name)| wgn::PipelineStageDescriptor {
                module: ps.module.id,
                stage: ps.stage,
                entry_point: ep_name.as_ptr(),
            })
            .collect::<ArrayVec<[_; 2]>>();

        RenderPipeline {
            id: wgn::wgpu_device_create_render_pipeline(self.id, wgn::RenderPipelineDescriptor {
                layout: desc.layout.id,
                stages: stages.as_ptr(),
                stages_length: stages.len(),
                primitive_topology: desc.primitive_topology,
                blend_state: desc.blend_state.as_ptr() as *const _,
                blend_state_length: desc.blend_state.len(),
                depth_stencil_state: desc.depth_stencil_state.id,
                attachment_state: desc.attachment_state.id,
            }),
        }
    }
}

impl Queue {
    pub fn submit(&self, command_buffers: &[CommandBuffer]) {
        wgn::wgpu_queue_submit(
            self.id,
            command_buffers.as_ptr() as *const _,
            command_buffers.len(),
        );
    }
}
