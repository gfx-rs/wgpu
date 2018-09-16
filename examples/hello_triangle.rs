extern crate gpu_native as gn;

fn main() {
    let instance = gn::create_instance();
    let adapter = gn::instance_get_adapter(instance, gn::AdapterDescriptor {
        power_preference: gn::PowerPreference::LowPower,
    });
    let device = gn::adapter_create_device(adapter, gn::DeviceDescriptor {
        extensions: gn::Extensions {
            anisotropic_filtering: false,
        },
    });
    let _vs = gn::device_create_shader_module(device, gn::ShaderModuleDescriptor {
        code: include_bytes!("./data/hello_triangle.vert.spv"),
    });
    let _fs = gn::device_create_shader_module(device, gn::ShaderModuleDescriptor {
        code: include_bytes!("./data/hello_triangle.frag.spv"),
    });
}
