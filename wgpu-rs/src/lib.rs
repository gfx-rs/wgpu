extern crate wgpu_native as wgn;

pub use wgn::{
    Color, Origin3d, Extent3d,
    AdapterDescriptor, Extensions, DeviceDescriptor, PowerPreference,
    ShaderModuleDescriptor, CommandBufferDescriptor,
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

pub struct ShaderModule {
    id: wgn::ShaderModuleId,
}

pub struct CommandBuffer {
    id: wgn::CommandBufferId,
}

pub struct Queue {
    id: wgn::QueueId,
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
        let desc = wgn::ShaderModuleDescriptor{
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
