use crate::core_tests::common;

#[test]
fn initialize() {
    let _ = wgpu::Instance::new(common::init::get_backend_bits());
}

fn request_adapter_inner(power: wgt::PowerPreference) {
    let instance = wgpu::Instance::new(common::init::get_backend_bits());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: power,
        compatible_surface: None,
    }))
    .unwrap();
}

#[test]
fn request_adapter_low_power() {
    request_adapter_inner(wgt::PowerPreference::LowPower);
}

#[test]
fn request_adapter_high_power() {
    request_adapter_inner(wgt::PowerPreference::HighPerformance);
}
