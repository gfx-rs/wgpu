//! Validation tests for the WGSL `unrestricted_pointer_parameters` language
//! extension.

use wgpu_test::valid;

#[test]
fn reported_in_wgsl_language_features() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    assert!(instance
        .wgsl_language_features()
        .contains(wgpu::WgslLanguageFeatures::UnrestrictedPointerParameters));
}

/// A `requires` directive only documents the use of a language extension, so
/// a pointer to part of a variable is accepted as an argument either way.
#[test]
fn sub_object_pointer_argument() {
    let (device, _queue) = wgpu::Device::noop(&Default::default());
    const BODY: &str = "
        struct R { m: mat2x2<f32> }
        struct Outer { inner: R }

        fn f(r: ptr<function, R>) {
            (*r).m = mat2x2<f32>();
        }

        @compute @workgroup_size(1)
        fn main() {
            var n: Outer;
            f(&n.inner);
        }
    ";
    for source in [
        format!("requires unrestricted_pointer_parameters;\n{BODY}"),
        BODY.to_string(),
    ] {
        valid(&device, || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        });
    }
}
