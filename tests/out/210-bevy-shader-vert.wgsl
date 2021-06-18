[[block]]
struct Camera {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct Transform {
    Model: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0), interpolate(perspective)]] v_Position: vec3<f32>;
    [[location(1), interpolate(perspective)]] v_Normal: vec3<f32>;
    [[location(2), interpolate(perspective)]] v_Uv: vec2<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

var<private> Vertex_Position1: vec3<f32>;
var<private> Vertex_Normal1: vec3<f32>;
var<private> Vertex_Uv1: vec2<f32>;
var<private> v_Position: vec3<f32>;
var<private> v_Normal: vec3<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: Camera;
[[group(2), binding(0)]]
var<uniform> global1: Transform;
var<private> gl_Position: vec4<f32>;

fn main1() {
    let _e10: mat4x4<f32> = global1.Model;
    let _e11: vec3<f32> = Vertex_Normal1;
    v_Normal = (_e10 * vec4<f32>(_e11, 1.0)).xyz;
    let _e16: mat4x4<f32> = global1.Model;
    let _e24: vec3<f32> = Vertex_Normal1;
    v_Normal = (mat3x3<f32>(_e16[0].xyz, _e16[1].xyz, _e16[2].xyz) * _e24);
    let _e26: mat4x4<f32> = global1.Model;
    let _e27: vec3<f32> = Vertex_Position1;
    v_Position = (_e26 * vec4<f32>(_e27, 1.0)).xyz;
    let _e32: vec2<f32> = Vertex_Uv1;
    v_Uv = _e32;
    let _e34: mat4x4<f32> = global.ViewProj;
    let _e35: vec3<f32> = v_Position;
    gl_Position = (_e34 * vec4<f32>(_e35, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0), interpolate(perspective)]] Vertex_Position: vec3<f32>, [[location(1), interpolate(perspective)]] Vertex_Normal: vec3<f32>, [[location(2), interpolate(perspective)]] Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position1 = Vertex_Position;
    Vertex_Normal1 = Vertex_Normal;
    Vertex_Uv1 = Vertex_Uv;
    main1();
    let _e7: vec3<f32> = v_Position;
    let _e9: vec3<f32> = v_Normal;
    let _e11: vec2<f32> = v_Uv;
    let _e13: vec4<f32> = gl_Position;
    return VertexOutput(_e7, _e9, _e11, _e13);
}
