struct VertexOutput {
    @location(0) v_uv: vec2<f32>,
    @builtin(position) member: vec4<f32>,
}

const c_scale: f32 = 1.2;

var<private> a_pos_1: vec2<f32>;
var<private> a_uv_1: vec2<f32>;
var<private> v_uv: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e4 = a_uv_1;
    v_uv = _e4;
    let _e6 = a_pos_1;
    let _e8 = (c_scale * _e6);
    gl_Position = vec4<f32>(_e8.x, _e8.y, 0.0, 1.0);
    return;
}

@vertex 
fn main(@location(0) a_pos: vec2<f32>, @location(1) a_uv: vec2<f32>) -> VertexOutput {
    a_pos_1 = a_pos;
    a_uv_1 = a_uv;
    main_1();
    let _e13 = v_uv;
    let _e15 = gl_Position;
    return VertexOutput(_e13, _e15);
}
