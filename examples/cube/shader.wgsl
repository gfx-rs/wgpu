[[location(0)]]
var<in> in_position: vec4<f32>;
[[location(1)]]
var<in> in_tex_coord_vs: vec2<f32>;
[[location(0)]]
var<out> out_tex_coord: vec2<f32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[block]]
struct Locals {
    transform: mat4x4<f32>;
};
[[group(0), binding(0)]]
var r_locals: Locals;

[[stage(vertex)]]
fn vs_main() {
    out_tex_coord = in_tex_coord_vs;
    out_position = r_locals.transform * in_position;
}

[[location(0)]]
var<in> in_tex_coord_fs: vec2<f32>;
[[location(0)]]
var<out> out_color: vec4<f32>;
[[group(0), binding(1)]]
var r_color: texture_2d<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main() {
    var tex: vec4<f32> = textureSample(r_color, r_sampler, in_tex_coord_fs);
    out_color = tex;
    //TODO: support `length` and `mix` functions
    //var mag: f32 = length(in_tex_coord_fs-vec2<f32>(0.5, 0.5));
    //out_color = vec4<f32>(mix(tex.xyz, vec3<f32>(0.0, 0.0, 0.0), mag*mag), 1.0);
}

[[stage(fragment)]]
fn fs_wire() {
    out_color = vec4<f32>(0.0, 0.5, 0.0, 0.5);
}
