[[builtin(position)]]
var<out> out_position: vec4<f32>;
[[location(0)]] var<out> out_uv: vec3<f32>;
[[builtin(vertex_index)]] var<in> in_vertex_index: u32;

[[block]]
struct Data {
    proj_inv: mat4x4<f32>;
    view: mat4x4<f32>;
};
[[group(0), binding(0)]]
var r_data: Data;

[[stage(vertex)]]
fn vs_main() {
    // hacky way to draw a large triangle
    var tmp1: i32 = i32(in_vertex_index) / 2;
    var tmp2: i32 = i32(in_vertex_index) & 1;
    const pos: vec4<f32> = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        0.0,
        1.0
    );

    const inv_model_view: mat3x3<f32> = transpose(mat3x3<f32>(r_data.view.x.xyz, r_data.view.y.xyz, r_data.view.z.xyz));
    var unprojected: vec4<f32> = r_data.proj_inv * pos; //TODO: const
    out_uv = inv_model_view * unprojected.xyz;
    out_position = pos;
}

[[group(0), binding(1)]]
var r_texture: texture_cube<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[location(0)]] var<in> in_uv: vec3<f32>;
[[location(0)]] var<out> out_color: vec4<f32>;

[[stage(fragment)]]
fn fs_main() {
    out_color = textureSample(r_texture, r_sampler, in_uv);
}
