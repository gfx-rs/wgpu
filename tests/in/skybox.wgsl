struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] uv: vec3<f32>;
};

[[block]]
struct Data {
    proj_inv: mat4x4<f32>;
    view: mat4x4<f32>;
};
[[group(0), binding(0)]]
var r_data: Data;

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] vertex_index: u32) -> VertexOutput {
    // hacky way to draw a large triangle
    var tmp1: i32 = i32(vertex_index) / 2;
    var tmp2: i32 = i32(vertex_index) & 1;
    const pos = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        0.0,
        1.0
    );

    const inv_model_view = transpose(mat3x3<f32>(r_data.view.x.xyz, r_data.view.y.xyz, r_data.view.z.xyz));
    const unprojected = r_data.proj_inv * pos;
    var out: VertexOutput;
    out.uv = inv_model_view * unprojected.xyz;
    out.position = pos;
    return out;
}

[[group(0), binding(1)]]
var r_texture: texture_cube<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_texture, r_sampler, in.uv);
}
