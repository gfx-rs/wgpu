struct VertexOutput {
    [[location(0), interpolate(perspective)]] v_uv: vec2<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

var<private> a_pos1: vec2<f32>;
var<private> a_uv1: vec2<f32>;
var<private> v_uv: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main1() {
    let _e4: vec2<f32> = a_uv1;
    v_uv = _e4;
    let _e6: vec2<f32> = a_pos1;
    gl_Position = vec4<f32>((1.2000000476837158 * _e6), 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main([[location(0), interpolate(perspective)]] a_pos: vec2<f32>, [[location(1), interpolate(perspective)]] a_uv: vec2<f32>) -> VertexOutput {
    a_pos1 = a_pos;
    a_uv1 = a_uv;
    main1();
    let _e5: vec2<f32> = v_uv;
    let _e7: vec4<f32> = gl_Position;
    return VertexOutput(_e5, _e7);
}
