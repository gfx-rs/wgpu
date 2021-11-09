[[block]]
struct gl_PerVertex {
    [[builtin(position)]] gl_Position: vec4<f32>;
};

struct VertexOutput {
    [[location(0)]] member: vec2<f32>;
    [[builtin(position)]] gl_Position: vec4<f32>;
};

var<private> v_uv: vec2<f32>;
var<private> a_uv_1: vec2<f32>;
var<private> perVertexStruct: gl_PerVertex = gl_PerVertex(vec4<f32>(0.0, 0.0, 0.0, 1.0), );
var<private> a_pos_1: vec2<f32>;

fn main_1() {
    let e12: vec2<f32> = a_uv_1;
    v_uv = e12;
    let e13: vec2<f32> = a_pos_1;
    perVertexStruct.gl_Position = vec4<f32>(e13.x, e13.y, 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main([[location(1)]] a_uv: vec2<f32>, [[location(0)]] a_pos: vec2<f32>) -> VertexOutput {
    a_uv_1 = a_uv;
    a_pos_1 = a_pos;
    main_1();
    let e7: vec2<f32> = v_uv;
    let e8: vec4<f32> = perVertexStruct.gl_Position;
    return VertexOutput(e7, e8);
}
