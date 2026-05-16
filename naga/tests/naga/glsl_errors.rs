//! Tests for known GLSL front-end limitations and error cases.
//!
//! Complements the snapshot tests in `snapshots.rs` by documenting parse/compile
//! failures that are expected given the current state of the GLSL frontend.

#![cfg(feature = "glsl-in")]

use naga::front::glsl::{Frontend, Options};

fn parse_frag(source: &str) -> Result<naga::Module, naga::front::glsl::ParseErrors> {
    Frontend::default().parse(
        &Options {
            stage: naga::ShaderStage::Fragment,
            defines: Default::default(),
        },
        source,
    )
}

fn parse_comp(source: &str) -> Result<naga::Module, naga::front::glsl::ParseErrors> {
    Frontend::default().parse(
        &Options {
            stage: naga::ShaderStage::Compute,
            defines: Default::default(),
        },
        source,
    )
}

/// `texture(sampler2D_uniform, uv)` — calling `texture()` directly on a `sampler2D`
/// uniform that was declared as a combined image+sampler type — is now supported.
///
/// When a `sampler2D` uniform is declared, naga synthesises a paired implicit sampler
/// global at the same binding.  `texture(u_tex, uv)` is then lowered to
/// `textureSample(u_tex, u_tex_sampler, uv)` using that implicit sampler.
///
/// The explicit constructor syntax `texture(sampler2D(u_tex, u_samp), uv)` with
/// separate `texture2D` + `sampler` uniforms (as in `tests/in/glsl/samplers.frag`)
/// continues to work unchanged.
///
/// The snapshot for this shader lives in `tests/in/glsl/sampler-combined-texture.frag`.
#[test]
fn texture_call_on_combined_sampler_uniform_is_unsupported() {
    let src = r#"
        #version 450
        layout(set = 0, binding = 0) uniform sampler2D u_tex;
        layout(location = 0) in vec2 v_uv;
        layout(location = 0) out vec4 o_color;
        void main() {
            o_color = texture(u_tex, v_uv);
        }
    "#;

    assert!(
        parse_frag(src).is_ok(),
        "expected texture(sampler2D_uniform, uv) to compile successfully"
    );
}

/// Shadow sampler variant: `texture(sampler2DShadow_uniform, coord)` is now supported.
///
/// When a `sampler2DShadow` uniform is declared, naga synthesises a paired implicit
/// comparison sampler at the same binding.  `texture(u_shadow, coord)` is then lowered
/// to `textureSampleCompare(u_shadow, u_shadow_sampler, coord.xy, coord.z)`.
#[test]
fn texture_call_on_shadow_sampler_uniform_is_unsupported() {
    let src = r#"
        #version 450
        layout(set = 0, binding = 0) uniform sampler2DShadow u_shadow;
        layout(location = 0) in vec3 v_coord;
        layout(location = 0) out float o_depth;
        void main() {
            o_depth = texture(u_shadow, v_coord);
        }
    "#;

    assert!(
        parse_frag(src).is_ok(),
        "expected texture(sampler2DShadow_uniform, coord) to compile successfully"
    );
}

/// imageLoad and imageStore in a compute shader work correctly.
/// This is a compile-only smoke test; the full snapshot lives in
/// `tests/in/glsl/image-compute.comp`.
#[test]
fn image_load_store_compute_compiles() {
    let src = r#"
        #version 460
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        layout(rgba8, set = 0, binding = 0) uniform readonly image2D src;
        layout(rgba8, set = 0, binding = 1) uniform writeonly image2D dst;
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            imageStore(dst, pos, imageLoad(src, pos));
        }
    "#;

    parse_comp(src).expect("imageLoad/imageStore in compute should compile");
}
