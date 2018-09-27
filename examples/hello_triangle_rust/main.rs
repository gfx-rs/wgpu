extern crate wgpu;
fn main() {
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(
        wgpu::AdapterDescriptor {
            power_preference: wgpu::PowerPreference::LowPower,
        },
    );
    let device = adapter.create_device(
        wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
        },
    );
    let vs_bytes = include_bytes!("./../data/hello_triangle.vert.spv");
    let _vs = device.create_shader_module(vs_bytes);
    let fs_bytes = include_bytes!("./../data/hello_triangle.frag.spv");
    let _fs = device.create_shader_module(fs_bytes);

    let cmd_buf = device.create_command_buffer(wgpu::CommandBufferDescriptor {
    });
    let queue = device.get_queue();
    queue.submit(&[cmd_buf]);
}
