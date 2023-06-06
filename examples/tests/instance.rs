use wasm_bindgen_test::*;

#[test]
#[wasm_bindgen_test]
fn initialize() {
    let _ = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all),
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
    });
}

fn request_adapter_inner(power: wgt::PowerPreference) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all),
        dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
    });

    let _adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: power,
        force_fallback_adapter: false,
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
