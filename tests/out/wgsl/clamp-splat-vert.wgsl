struct VertexOutput {
    [[builtin(position)]] member: vec4<f32>;
};

var<private> a_pos_1: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let e5: vec2<f32> = a_pos_1;
    gl_Position = vec4<f32>(clamp(e5, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] a_pos: vec2<f32>) -> VertexOutput {
    a_pos_1 = a_pos;
    main_1();
    let e5: vec4<f32> = gl_Position;
    return VertexOutput(e5);
}
