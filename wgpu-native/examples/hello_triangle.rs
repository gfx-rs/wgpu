extern crate wgpu_native as wgn;

fn main() {
    let instance = wgn::create_instance();
    let adapter = wgn::instance_get_adapter(instance, wgn::AdapterDescriptor {
        power_preference: wgn::PowerPreference::LowPower,
    });
    let device = wgn::adapter_create_device(adapter, wgn::DeviceDescriptor {
        extensions: wgn::Extensions {
            anisotropic_filtering: false,
        },
    });
    let _vs = wgn::device_create_shader_module(device, wgn::ShaderModuleDescriptor {
        code: include_bytes!("./data/hello_triangle.vert.spv"),
    });
    let _fs = wgn::device_create_shader_module(device, wgn::ShaderModuleDescriptor {
        code: include_bytes!("./data/hello_triangle.frag.spv"),
    });
}
