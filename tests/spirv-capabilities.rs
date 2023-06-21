/*!
Test SPIR-V backend capability checks.
*/

#![cfg(all(feature = "wgsl-in", feature = "spv-out"))]

use spirv::Capability as Ca;

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
    writer.write(&module, &info, None, &mut words).unwrap();
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
        panic!(
            "shader code should have requested these caps: {:?}\n\n{}",
            missing_caps, source
        );
    }

    let forbidden_caps: Vec<_> = forbidden
        .iter()
        .filter(|&cap| caps_used.contains(cap))
        .cloned()
        .collect();
    if !forbidden_caps.is_empty() {
        panic!(
            "shader code should not have requested these caps: {:?}\n\n{}",
            forbidden_caps, source
        );
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
        fn f(i: texture_2d<f32>) -> vec2<i32> {
            return textureDimensions(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_2d_array<f32>) -> i32 {
            return textureNumLayers(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_2d<f32>) -> i32 {
            return textureNumLevels(i);
        }
    "#,
    );
    require(
        &[Ca::ImageQuery],
        r#"
        fn f(i: texture_multisampled_2d<f32>) -> i32 {
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
