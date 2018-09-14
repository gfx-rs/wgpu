extern crate gpu_native as gn;

fn main() {
    let instance = gn::create_instance();
    let adapter = gn::instance_get_adapter(instance, gn::AdapterDescriptor {
        power_preference: gn::PowerPreference::LowPower,
    });
    let device = gn::adapter_create_device(adapter, gn::DeviceDescriptor {
        extensions: gn::Extensions {
            anisotropicFiltering: false,
        },
    });
}
