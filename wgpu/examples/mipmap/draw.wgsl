[[location(0)]]
var<in> in_position_vs: vec4<f32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;
[[location(0)]]
var<out> out_tex_coords_vs: vec2<f32>;

[[block]]
struct Locals {
    transform: mat4x4<f32>;
};
[[group(0), binding(0)]]
var r_data: Locals;

[[stage(vertex)]]
fn vs_main() {
    out_tex_coords_vs = 0.05 * in_position_vs.xy + vec2<f32>(0.5, 0.5);
    out_position = r_data.transform * in_position_vs;
}

[[location(0)]]
var<in> in_tex_coord_fs: vec2<f32>;
[[location(0)]]
var<out> out_color_fs: vec4<f32>;

[[group(0), binding(1)]]
var r_color: texture_2d<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main() {
    out_color_fs = textureSample(r_color, r_sampler, in_tex_coord_fs);
}
