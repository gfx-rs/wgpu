[[block]]
struct gl_PerVertex {
    [[builtin(position)]] gl_Position: vec4<f32>;
};

struct VertexOutput {
    [[location(0), interpolate(perspective)]] member: vec2<f32>;
    [[builtin(position)]] gl_Position1: vec4<f32>;
};

var<private> v_uv: vec2<f32>;
var<private> a_uv: vec2<f32>;
var<private> perVertexStruct: gl_PerVertex;
var<private> a_pos: vec2<f32>;

fn main() {
    let _e12: vec2<f32> = a_uv;
    v_uv = _e12;
    let _e13: vec2<f32> = a_pos;
    perVertexStruct.gl_Position = vec4<f32>(_e13.x, _e13.y, 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main1([[location(1)]] a_uv1: vec2<f32>, [[location(0)]] a_pos1: vec2<f32>) -> VertexOutput {
    a_uv = a_uv1;
    a_pos = a_pos1;
    main();
    let _e10: vec2<f32> = v_uv;
    let _e11: vec4<f32> = perVertexStruct.gl_Position;
    return VertexOutput(_e10, _e11);
}
