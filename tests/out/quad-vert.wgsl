[[block]]
struct gl_PerVertex {
    [[builtin(position)]] gl_Position: vec4<f32>;
};

struct type10 {
    [[location(0), interpolate(perspective)]] member: vec2<f32>;
    [[builtin(position)]] gl_Position1: vec4<f32>;
};

var v_uv: vec2<f32>;
var a_uv: vec2<f32>;
var perVertexStruct: gl_PerVertex;
var a_pos: vec2<f32>;

fn main() {
    v_uv = a_uv;
    let _e13: vec2<f32> = a_pos;
    perVertexStruct.gl_Position = vec4<f32>(_e13[0], _e13[1], 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main1([[location(1)]] a_uv1: vec2<f32>, [[location(0)]] a_pos1: vec2<f32>) -> type10 {
    a_uv = a_uv1;
    a_pos = a_pos1;
    main();
    return type10(v_uv, perVertexStruct.gl_Position);
}
