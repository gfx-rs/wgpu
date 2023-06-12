struct VertexOutput {
    @builtin(position) member: vec4<f32>,
}

var<private> a_pos_1: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e5 = a_pos_1;
    let _e10 = clamp(_e5, vec2<f32>(0.0), vec2<f32>(1.0));
    gl_Position = vec4<f32>(_e10.x, _e10.y, 0.0, 1.0);
    return;
}

@vertex 
fn main(@location(0) a_pos: vec2<f32>) -> VertexOutput {
    a_pos_1 = a_pos;
    main_1();
    let _e5 = gl_Position;
    return VertexOutput(_e5);
}
