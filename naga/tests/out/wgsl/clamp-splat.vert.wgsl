struct VertexOutput {
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> a_pos_1: vec2<f32>;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e2 = a_pos_1;
    let _e7 = clamp(_e2, vec2(0f), vec2(1f));
    gl_Position = vec4<f32>(_e7.x, _e7.y, 0f, 1f);
    return;
}

@vertex 
fn main(@location(0) a_pos: vec2<f32>) -> VertexOutput {
    a_pos_1 = a_pos;
    main_1();
    let _e5 = gl_Position;
    return VertexOutput(_e5);
}
