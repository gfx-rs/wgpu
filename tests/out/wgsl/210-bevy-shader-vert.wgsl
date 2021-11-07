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
    let e10: mat4x4<f32> = global_1.Model;
    let e11: vec3<f32> = Vertex_Normal_1;
    v_Normal = (e10 * vec4<f32>(e11, 1.0)).xyz;
    let e16: mat4x4<f32> = global_1.Model;
    let e26: vec3<f32> = Vertex_Normal_1;
    v_Normal = (mat3x3<f32>(e16[0].xyz, e16[1].xyz, e16[2].xyz) * e26);
    let e28: mat4x4<f32> = global_1.Model;
    let e29: vec3<f32> = Vertex_Position_1;
    v_Position = (e28 * vec4<f32>(e29, 1.0)).xyz;
    let e34: vec2<f32> = Vertex_Uv_1;
    v_Uv = e34;
    let e36: mat4x4<f32> = global.ViewProj;
    let e37: vec3<f32> = v_Position;
    gl_Position = (e36 * vec4<f32>(e37, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(1)]] Vertex_Normal: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let e23: vec3<f32> = v_Position;
    let e25: vec3<f32> = v_Normal;
    let e27: vec2<f32> = v_Uv;
    let e29: vec4<f32> = gl_Position;
    return VertexOutput(e23, e25, e27, e29);
}
