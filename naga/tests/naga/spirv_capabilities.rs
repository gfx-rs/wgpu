/*!
Test SPIR-V backend capability checks.
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use spirv::Capability as Ca;

#[cfg(spv_out)]
use rspirv::binary::Disassemble;

fn capabilities_used(source: &str) -> naga::FastIndexSet<Ca> {
    use naga::back::spv;
    use naga::valid;

    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!(
            "expected WGSL to parse successfully:\n{}",
            e.emit_to_string(source)
        );
    });

    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validation failed");

    let mut words = vec![];
    let mut writer = spv::Writer::new(&spv::Options::default()).unwrap();
    writer
        .write(&module, &info, None, &None, &mut words)
        .unwrap();
    writer.get_capabilities_used().clone()
}

fn require(capabilities: &[Ca], source: &str) {
    require_and_forbid(capabilities, &[], source);
}

fn require_and_forbid(required: &[Ca], forbidden: &[Ca], source: &str) {
    let caps_used = capabilities_used(source);

    let missing_caps: Vec<_> = required
        .iter()
        .filter(|&cap| !caps_used.contains(cap))
        .cloned()
        .collect();
    if !missing_caps.is_empty() {
        panic!("shader code should have requested these caps: {missing_caps:?}\n\n{source}");
    }

    let forbidden_caps: Vec<_> = forbidden
        .iter()
        .filter(|&cap| caps_used.contains(cap))
        .cloned()
        .collect();
    if !forbidden_caps.is_empty() {
        panic!("shader code should not have requested these caps: {forbidden_caps:?}\n\n{source}");
    }
}

#[test]
fn sampler1d() {
    require(
        &[Ca::Sampled1D],
        r#"
        @group(0) @binding(0)
        var image_1d: texture_1d<f32>;
    "#,
    );
}

#[test]
fn storage1d() {
    require(
        &[Ca::Image1D],
        r#"
        @group(0) @binding(0)
        var image_1d: texture_storage_1d<rgba8unorm,write>;
    "#,
    );
}

#[test]
fn cube_array() {
    // ImageCubeArray is only for storage cube array images, which WGSL doesn't
    // support
    require_and_forbid(
        &[Ca::SampledCubeArray],
        &[Ca::ImageCubeArray],
        r#"
        @group(0) @binding(0)
        var image_cube: texture_cube_array<f32>;
    "#,
    );
}

#[test]
fn image_queries() {
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_2d<f32>) -> vec2<u32> {
            return textureDimensions(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_2d_array<f32>) -> u32 {
            return textureNumLayers(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_2d<f32>) -> u32 {
            return textureNumLevels(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_multisampled_2d<f32>) -> u32 {
            return textureNumSamples(i);
        }
    "#,
    );
}

#[test]
fn sample_rate_shading() {
    require(
        &[Ca::SampleRateShading],
        r#"
        @fragment
        fn f(@location(0) @interpolate(perspective, sample) x: f32) { }
    "#,
    );

    require(
        &[Ca::SampleRateShading],
        r#"
        @fragment
        fn f(@builtin(sample_index) x: u32) { }
    "#,
    );
}

#[test]
fn geometry() {
    require(
        &[Ca::Geometry],
        r#"
        @fragment
        fn f(@builtin(primitive_index) x: u32) { }
    "#,
    );
}

#[test]
fn storage_image_formats() {
    require_and_forbid(
        &[Ca::Shader],
        &[Ca::StorageImageExtendedFormats],
        r#"
            @group(0) @binding(0)
            var image_rg32f: texture_storage_2d<rgba16uint, read>;
        "#,
    );

    require(
        &[Ca::StorageImageExtendedFormats],
        r#"
            @group(0) @binding(0)
            var image_rg32f: texture_storage_2d<rg32float, read>;
        "#,
    );
}

#[test]
fn float64() {
    require(
        &[Ca::Float64],
        r#"
            fn f(x: f64) -> f64 {
                return x;
            }
        "#,
    );
}

#[test]
fn int64() {
    require(
        &[Ca::Int64],
        r#"
            fn f(x: i64) -> i64 {
                return x;
            }
        "#,
    );
    require(
        &[Ca::Int64],
        r#"
            fn f(x: u64) -> u64 {
                return x;
            }
        "#,
    );
}

#[test]
fn float16() {
    require(&[Ca::Float16], "enable f16; fn f(x: f16) { }");
}

#[test]
fn f16_io_capabilities() {
    let source = r#"
        enable f16;
        
        struct VertexOutput {
            @location(0) color: vec3<f16>,
        }
        
        @fragment  
        fn main(input: VertexOutput) -> @location(0) vec4<f16> {
            return vec4<f16>(input.color, f16(1.0));
        }
    "#;

    use naga::back::spv;
    use naga::valid;

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    // Test native path: use_storage_input_output_16 = true
    let options_native = spv::Options {
        use_storage_input_output_16: true,
        ..Default::default()
    };

    let mut words_native = vec![];
    let mut writer_native = spv::Writer::new(&options_native).unwrap();
    writer_native
        .write(&module, &info, None, &None, &mut words_native)
        .unwrap();
    let caps_native = writer_native.get_capabilities_used();

    // Should include `StorageInputOutput16` for native `f16` I/O
    assert!(caps_native.contains(&Ca::StorageInputOutput16));

    // Test polyfill path: use_storage_input_output_16 = false
    let options_polyfill = spv::Options {
        use_storage_input_output_16: false,
        ..Default::default()
    };

    let mut words_polyfill = vec![];
    let mut writer_polyfill = spv::Writer::new(&options_polyfill).unwrap();
    writer_polyfill
        .write(&module, &info, None, &None, &mut words_polyfill)
        .unwrap();
    let caps_polyfill = writer_polyfill.get_capabilities_used();

    // Should not include `StorageInputOutput16` when polyfilled
    assert!(!caps_polyfill.contains(&Ca::StorageInputOutput16));

    // But should still include the basic `f16` capabilities
    assert!(caps_polyfill.contains(&Ca::Float16));
}

#[cfg(spv_out)]
#[test]
fn f16_io_polyfill_codegen() {
    let source = r#"
        enable f16;

        struct F16IO {
            @location(0) scalar_f16: f16,
            @location(1) scalar_f32: f32,
            @location(2) vec2_f16: vec2<f16>,
            @location(3) vec2_f32: vec2<f32>,
        }

        @fragment
        fn main(input: F16IO) -> F16IO {
            var output = input;
            output.scalar_f16 = input.scalar_f16 + 1.0h;
            output.vec2_f16.x = input.vec2_f16.y;
            return output;
        }
    "#;

    use naga::{back::spv, valid};

    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .unwrap();

    // Test Native Path
    let options_native = spv::Options {
        use_storage_input_output_16: true,
        ..Default::default()
    };
    let mut words_native = vec![];
    let mut writer_native = spv::Writer::new(&options_native).unwrap();
    writer_native
        .write(&module, &info, None, &None, &mut words_native)
        .unwrap();
    let caps_native = writer_native.get_capabilities_used();
    let dis_native = rspirv::dr::load_words(words_native).unwrap().disassemble();

    // Native path must request the capability and must NOT have conversions.
    assert!(caps_native.contains(&Ca::StorageInputOutput16));
    assert!(!dis_native.contains("OpFConvert"));

    // Test Polyfill Path
    let options_polyfill = spv::Options {
        use_storage_input_output_16: false,
        ..Default::default()
    };
    let mut words_polyfill = vec![];
    let mut writer_polyfill = spv::Writer::new(&options_polyfill).unwrap();
    writer_polyfill
        .write(&module, &info, None, &None, &mut words_polyfill)
        .unwrap();
    let caps_polyfill = writer_polyfill.get_capabilities_used();
    let dis_polyfill = rspirv::dr::load_words(words_polyfill)
        .unwrap()
        .disassemble();

    // Polyfill path should request the capability but not have conversions.
    assert!(!caps_polyfill.contains(&Ca::StorageInputOutput16));
    assert!(dis_polyfill.contains("OpFConvert"));

    // Should have 2 input conversions, and 2 output conversions
    let fconvert_count = dis_polyfill.matches("OpFConvert").count();
    assert_eq!(
        fconvert_count, 4,
        "Expected 4 OpFConvert instructions for polyfilled I/O"
    );
}
