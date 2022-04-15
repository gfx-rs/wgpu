struct Camera {
    ViewProj: mat4x4<f32>,
}

struct Transform {
    Model: mat4x4<f32>,
}

struct VertexOutput {
    @location(0) v_Position: vec3<f32>,
    @location(1) v_Normal: vec3<f32>,
    @location(2) v_Uv: vec2<f32>,
    @builtin(position) member: vec4<f32>,
}

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> v_Position: vec3<f32>;
var<private> v_Normal: vec3<f32>;
var<private> v_Uv: vec2<f32>;
@group(0) @binding(0) 
var<uniform> global: Camera;
@group(2) @binding(0) 
var<uniform> global_1: Transform;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    let _e10 = global_1.Model;
    let _e11 = Vertex_Normal_1;
    v_Normal = (_e10 * vec4<f32>(_e11.x, _e11.y, _e11.z, 1.0)).xyz;
    let _e19 = global_1.Model;
    let _e29 = Vertex_Normal_1;
    v_Normal = (mat3x3<f32>(_e19[0].xyz, _e19[1].xyz, _e19[2].xyz) * _e29);
    let _e31 = global_1.Model;
    let _e32 = Vertex_Position_1;
    v_Position = (_e31 * vec4<f32>(_e32.x, _e32.y, _e32.z, 1.0)).xyz;
    let _e40 = Vertex_Uv_1;
    v_Uv = _e40;
    let _e42 = global.ViewProj;
    let _e43 = v_Position;
    gl_Position = (_e42 * vec4<f32>(_e43.x, _e43.y, _e43.z, 1.0));
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>) -> VertexOutput {
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
