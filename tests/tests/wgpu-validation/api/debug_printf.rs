use std::borrow::Cow;

use wgpu_test::{fail, valid};

const DEBUG_PRINTF_SHADER: &str = r#"
enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    debugPrintf("debug value: %d", 1i);
}
"#;

const DEBUG_PRINTF_WITHOUT_ENABLE_SHADER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    debugPrintf("debug value: %d", 1i);
}
"#;

const STRING_LITERAL_OUTSIDE_DEBUG_PRINTF_SHADER: &str = r#"
enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main() {
    let _value = "not a debugPrintf format";
}
"#;

fn debug_printf_device() -> wgpu::Device {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::DEBUG_PRINTF,
        ..Default::default()
    });
    device
}

#[test]
fn shader_module() {
    let device = debug_printf_device();
    valid(&device, || {
        create_shader_module(&device, DEBUG_PRINTF_SHADER);
    });
}

#[test]
fn requires_feature() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    fail(
        &device,
        || create_shader_module(&device, DEBUG_PRINTF_SHADER),
        Some("DEBUG_PRINTF"),
    );
}

#[test]
fn requires_enable_extension() {
    let device = debug_printf_device();
    fail(
        &device,
        || create_shader_module(&device, DEBUG_PRINTF_WITHOUT_ENABLE_SHADER),
        Some("enable extension is not enabled"),
    );
}

#[test]
fn rejects_string_literal_outside_call() {
    let device = debug_printf_device();
    fail(
        &device,
        || create_shader_module(&device, STRING_LITERAL_OUTSIDE_DEBUG_PRINTF_SHADER),
        Some("String literals are only supported in debugPrintf"),
    );
}

fn create_shader_module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("debugPrintf shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}
