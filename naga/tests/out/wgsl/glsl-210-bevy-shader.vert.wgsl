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
    @builtin(position) gl_Position: vec4<f32>,
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
    v_Normal = (_e10 * vec4<f32>(_e11.x, _e11.y, _e11.z, 1f)).xyz;
    let _e19 = global_1.Model;
    let _e27 = Vertex_Normal_1;
    v_Normal = (mat3x3<f32>(_e19[0].xyz, _e19[1].xyz, _e19[2].xyz) * _e27);
    let _e29 = global_1.Model;
    let _e30 = Vertex_Position_1;
    v_Position = (_e29 * vec4<f32>(_e30.x, _e30.y, _e30.z, 1f)).xyz;
    let _e38 = Vertex_Uv_1;
    v_Uv = _e38;
    let _e40 = global.ViewProj;
    let _e41 = v_Position;
    gl_Position = (_e40 * vec4<f32>(_e41.x, _e41.y, _e41.z, 1f));
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let _e7 = v_Position;
    let _e9 = v_Normal;
    let _e11 = v_Uv;
    let _e13 = gl_Position;
    return VertexOutput(_e7, _e9, _e11, _e13);
}
