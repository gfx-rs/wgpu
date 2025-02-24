struct CameraViewProj {
    ViewProj: mat4x4<f32>,
}

struct Transform {
    Model: mat4x4<f32>,
}

struct VertexOutput {
    @location(0) v_WorldPosition: vec3<f32>,
    @location(1) v_WorldNormal: vec3<f32>,
    @location(2) v_Uv: vec2<f32>,
    @location(3) v_WorldTangent: vec4<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> Vertex_Tangent_1: vec4<f32>;
var<private> v_WorldPosition: vec3<f32>;
var<private> v_WorldNormal: vec3<f32>;
var<private> v_Uv: vec2<f32>;
@group(0) @binding(0) 
var<uniform> global: CameraViewProj;
var<private> v_WorldTangent: vec4<f32>;
@group(2) @binding(0) 
var<uniform> global_1: Transform;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var world_position: vec4<f32>;

    let _e12 = global_1.Model;
    let _e13 = Vertex_Position_1;
    world_position = (_e12 * vec4<f32>(_e13.x, _e13.y, _e13.z, 1f));
    let _e21 = world_position;
    v_WorldPosition = _e21.xyz;
    let _e23 = global_1.Model;
    let _e31 = Vertex_Normal_1;
    v_WorldNormal = (mat3x3<f32>(_e23[0].xyz, _e23[1].xyz, _e23[2].xyz) * _e31);
    let _e33 = Vertex_Uv_1;
    v_Uv = _e33;
    let _e34 = global_1.Model;
    let _e42 = Vertex_Tangent_1;
    let _e44 = (mat3x3<f32>(_e34[0].xyz, _e34[1].xyz, _e34[2].xyz) * _e42.xyz);
    let _e45 = Vertex_Tangent_1;
    v_WorldTangent = vec4<f32>(_e44.x, _e44.y, _e44.z, _e45.w);
    let _e52 = global.ViewProj;
    let _e53 = world_position;
    gl_Position = (_e52 * _e53);
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>, @location(3) Vertex_Tangent: vec4<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    Vertex_Tangent_1 = Vertex_Tangent;
    main_1();
    let _e9 = v_WorldPosition;
    let _e11 = v_WorldNormal;
    let _e13 = v_Uv;
    let _e15 = v_WorldTangent;
    let _e17 = gl_Position;
    return VertexOutput(_e9, _e11, _e13, _e15, _e17);
}
