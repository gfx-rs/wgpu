struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

struct VertexOutput {
    @location(0) member: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> v_uv: vec2<f32>;
var<private> a_uv_1: vec2<f32>;
var<private> perVertexStruct: gl_PerVertex = gl_PerVertex(vec4<f32>(0.0, 0.0, 0.0, 1.0), 1.0, array<f32, 1>(), array<f32, 1>());
var<private> a_pos_1: vec2<f32>;

fn main_1() {
    let _e6 = a_uv_1;
    v_uv = _e6;
    let _e7 = a_pos_1;
    perVertexStruct.gl_Position = vec4<f32>(_e7.x, _e7.y, 0.0, 1.0);
    return;
}

@vertex 
fn main(@location(1) a_uv: vec2<f32>, @location(0) a_pos: vec2<f32>) -> VertexOutput {
    a_uv_1 = a_uv;
    a_pos_1 = a_pos;
    main_1();
    let _e7 = v_uv;
    let _e8 = perVertexStruct.gl_Position;
    return VertexOutput(_e7, _e8);
}
