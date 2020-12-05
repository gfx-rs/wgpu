[[builtin(vertex_index)]]
var<in> in_vertex_index: u32;
[[builtin(position)]]
var<out> out_pos: vec4<f32>;

[[stage(vertex)]]
fn vs_main() {
    # hacky way to draw a large triangle
    var tmp1: i32 = i32(in_vertex_index) / 2;
    var tmp2: i32 = i32(in_vertex_index) & 1;
    var pos: vec2<f32> = vec2<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0
    );
    out_pos = vec4<f32>(pos, 0.0, 1.0);
}

[[location(0)]]
var<out> out_color: vec4<f32>;

[[stage(fragment)]]
fn fs_main() {
    out_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
