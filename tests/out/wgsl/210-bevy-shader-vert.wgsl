[[block]]
struct Camera {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct Transform {
    Model: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0)]] v_Position: vec3<f32>;
    [[location(1)]] v_Normal: vec3<f32>;
    [[location(2)]] v_Uv: vec2<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> v_Position: vec3<f32>;
var<private> v_Normal: vec3<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: Camera;
[[group(2), binding(0)]]
var<uniform> global_1: Transform;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e10 = global_1.Model;
    let _e11 = Vertex_Normal_1;
    v_Normal = (_e10 * vec4<f32>(_e11, 1.0)).xyz;
    let _e16 = global_1.Model;
    let _e26 = Vertex_Normal_1;
    v_Normal = (mat3x3<f32>(_e16[0].xyz, _e16[1].xyz, _e16[2].xyz) * _e26);
    let _e28 = global_1.Model;
    let _e29 = Vertex_Position_1;
    v_Position = (_e28 * vec4<f32>(_e29, 1.0)).xyz;
    let _e34 = Vertex_Uv_1;
    v_Uv = _e34;
    let _e36 = global.ViewProj;
    let _e37 = v_Position;
    gl_Position = (_e36 * vec4<f32>(_e37, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(1)]] Vertex_Normal: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let _e23 = v_Position;
    let _e25 = v_Normal;
    let _e27 = v_Uv;
    let _e29 = gl_Position;
    return VertexOutput(_e23, _e25, _e27, _e29);
}
