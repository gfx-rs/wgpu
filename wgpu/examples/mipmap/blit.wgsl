[[builtin(vertex_index)]]
var<in> in_vertex_index: u32;
[[builtin(position)]]
var<out> out_position: vec4<f32>;
[[location(0)]]
var<out> out_tex_coords_vs: vec2<f32>;

[[stage(vertex)]]
fn vs_main() {
    var x: i32 = i32(in_vertex_index) / 2;
    var y: i32 = i32(in_vertex_index) & 1;
    out_tex_coords_vs = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    out_position = vec4<f32>(
        out_tex_coords_vs.x * 2.0 - 1.0,
        1.0 - out_tex_coords_vs.y * 2.0,
        0.0, 1.0
    );
}

[[location(0)]]
var<in> in_tex_coord_fs: vec2<f32>;
[[location(0)]]
var<out> out_color_fs: vec4<f32>;

[[group(0), binding(0)]]
var r_color: texture_2d<f32>;
[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main() {
    out_color_fs = textureSample(r_color, r_sampler, in_tex_coord_fs);
}
