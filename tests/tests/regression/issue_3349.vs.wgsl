@group(0) @binding(0)
var<uniform> data1: vec4f;

// D3DCompile requires this to be a struct
struct Pc {
    inner: vec4f,
}

var<push_constant> data2: Pc;

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) data1: vec4f,
    @location(1) data2: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VsOut {
    let uv = vec2f(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));
    let position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return VsOut(position, data1, data2.inner);
}
