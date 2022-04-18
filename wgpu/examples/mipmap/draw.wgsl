struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct Locals {
    transform: mat4x4<f32>
};
@group(0)
@binding(0)
var<uniform> r_data: Locals;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let pos = vec2<f32>(
        100.0 * (1.0 - f32(vertex_index & 2u)),
        1000.0 * f32(vertex_index & 1u)
    );
    var result: VertexOutput;
    result.tex_coords = 0.05 * pos + vec2<f32>(0.5, 0.5);
    result.position = r_data.transform * vec4<f32>(pos, 0.0, 1.0);
    return result;
}

@group(0)
@binding(1)
var r_color: texture_2d<f32>;
@group(0)
@binding(2)
var r_sampler: sampler;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(r_color, r_sampler, vertex.tex_coords);
}
