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
var<uniform> r_data: Data;
[[group(0), binding(1)]]
var r_texture: texture_cube<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] vertex_index: u32) -> VertexOutput {
    var tmp1: i32;
    var tmp2: i32;

    tmp1 = (i32(vertex_index) / 2);
    tmp2 = (i32(vertex_index) & 1);
    let e10: i32 = tmp1;
    let e16: i32 = tmp2;
    let pos: vec4<f32> = vec4<f32>(((f32(e10) * 4.0) - 1.0), ((f32(e16) * 4.0) - 1.0), 0.0, 1.0);
    let e27: vec4<f32> = r_data.view[0];
    let e31: vec4<f32> = r_data.view[1];
    let e35: vec4<f32> = r_data.view[2];
    let inv_model_view: mat3x3<f32> = transpose(mat3x3<f32>(e27.xyz, e31.xyz, e35.xyz));
    let e40: mat4x4<f32> = r_data.proj_inv;
    let unprojected: vec4<f32> = (e40 * pos);
    return VertexOutput(pos, (inv_model_view * unprojected.xyz));
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let e5: vec4<f32> = textureSample(r_texture, r_sampler, in.uv);
    return e5;
}
