//! Tests of the HLSL backend.

#![cfg(all(feature = "wgsl-in", feature = "hlsl-out"))]

use naga::back::hlsl;

/// Translate `source` for `entry_point`, with a binding map that mimics what
/// the `dx12` backend builds for a bind group layout holding a single uniform
/// buffer at `@binding(0)` and a texture at `@binding(1)`, and therefore no
/// sampler index buffer for group 0.
fn write_with_texture_at_binding_one(
    source: &str,
    entry_point: &str,
) -> Result<hlsl::ReflectionInfo, hlsl::Error> {
    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("module should validate");

    let mut binding_map = hlsl::BindingMap::default();
    binding_map.insert(
        naga::ResourceBinding {
            group: 0,
            binding: 0,
        },
        hlsl::BindTarget {
            space: 0,
            register: 0,
            ..Default::default()
        },
    );
    binding_map.insert(
        naga::ResourceBinding {
            group: 0,
            binding: 1,
        },
        hlsl::BindTarget {
            space: 0,
            register: 0,
            ..Default::default()
        },
    );

    let options = hlsl::Options {
        fake_missing_bindings: false,
        binding_map,
        sampler_buffer_binding_map: hlsl::SamplerIndexBufferBindingMap::default(),
        ..Default::default()
    };
    let pipeline_options = hlsl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Fragment, entry_point.to_string())),
    };

    let mut hlsl_source = String::new();
    let mut writer = hlsl::Writer::new(&mut hlsl_source, &options, &pipeline_options);
    writer.write(&module, &info, None)
}

/// A sampler global that the entry point never touches must not abort
/// translation when its bind group has no sampler index buffer.
///
/// <https://github.com/gfx-rs/wgpu/issues/7638>
#[test]
fn unused_sampler_without_sampler_index_buffer() {
    let reflection_info = write_with_texture_at_binding_one(
        "
        @group(0) @binding(0) var<uniform> u: vec4<f32>;
        @group(0) @binding(1) var s: sampler;

        @fragment
        fn main() -> @location(0) vec4<f32> {
            return u;
        }
        ",
        "main",
    )
    .expect("translation should succeed");

    assert_eq!(reflection_info.entry_point_names.len(), 1);
    reflection_info.entry_point_names[0]
        .as_ref()
        .expect("entry point should translate");
}

/// A sampler global that the entry point *does* use, whose bind group has no
/// sampler index buffer, must report a translation error for that entry point
/// rather than aborting the whole translation.
///
/// The declaration is skipped in this case too, so emitting the entry point
/// would produce HLSL referring to an undeclared name. The check in
/// `Writer::write` has to catch it first.
///
/// <https://github.com/gfx-rs/wgpu/issues/7638>
#[test]
fn used_sampler_without_sampler_index_buffer() {
    let reflection_info = write_with_texture_at_binding_one(
        "
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;

        @fragment
        fn main() -> @location(0) vec4<f32> {
            return textureSampleLevel(t, s, vec2<f32>(0.0), 0.0);
        }
        ",
        "main",
    )
    .expect("translation should succeed");

    assert_eq!(reflection_info.entry_point_names.len(), 1);
    assert!(matches!(
        reflection_info.entry_point_names[0],
        Err(hlsl::EntryPointError::MissingSamplerIndexBuffer(0))
    ));
}
