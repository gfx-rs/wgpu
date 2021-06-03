struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] vertex_index: u32) -> VertexOutput {
    let x: f32 = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y: f32 = f32(i32(vertex_index & 2u) << 1u) - 1.0;
    var output: VertexOutput;
    output.position = vec4<f32>(x, -y, 0.0, 1.0);
    output.tex_coords = vec2<f32>(x + 1.0, y + 1.0) * 0.5;
    return output;
}

[[group(0), binding(0)]]
var r_color: texture_2d<f32>;
[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_color, r_sampler, in.tex_coords);
}