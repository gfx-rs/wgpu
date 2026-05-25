// Regression test for https://github.com/gfx-rs/wgpu/issues/9115
//
// On the GLES backend, using immediates in both vertex and fragment shaders
// from a single shader module while also having a uniform buffer causes a panic:
//   "Unsupported uniform datatype: Struct { ... }"
// The backend confuses the uniform struct type with the immediates struct.

struct Globals {
    inv_screen_size: vec2f,
}
@group(0) @binding(0)
var<uniform> globals: Globals;

struct Immediates {
    position: vec2f,
    size: vec2f,
    color: vec4f,
}
var<immediate> immediates: Immediates;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    // Reference both globals and immediates in the vertex shader.
    _ = globals.inv_screen_size;
    _ = immediates.position + immediates.size;

    let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    // Accessing immediates in the fragment shader is what triggers the GLES panic.
    return immediates.color;
}
