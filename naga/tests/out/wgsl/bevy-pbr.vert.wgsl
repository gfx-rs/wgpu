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
    @builtin(position) member: vec4<f32>,
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
    world_position = (_e12 * vec4<f32>(_e13.x, _e13.y, _e13.z, 1.0));
    let _e21 = world_position;
    v_WorldPosition = _e21.xyz;
    let _e23 = global_1.Model;
    let _e33 = Vertex_Normal_1;
    v_WorldNormal = (mat3x3<f32>(_e23[0].xyz, _e23[1].xyz, _e23[2].xyz) * _e33);
    let _e35 = Vertex_Uv_1;
    v_Uv = _e35;
    let _e36 = global_1.Model;
    let _e46 = Vertex_Tangent_1;
    let _e48 = (mat3x3<f32>(_e36[0].xyz, _e36[1].xyz, _e36[2].xyz) * _e46.xyz);
    let _e49 = Vertex_Tangent_1;
    v_WorldTangent = vec4<f32>(_e48.x, _e48.y, _e48.z, _e49.w);
    let _e56 = global.ViewProj;
    let _e57 = world_position;
    gl_Position = (_e56 * _e57);
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>, @location(3) Vertex_Tangent: vec4<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    Vertex_Tangent_1 = Vertex_Tangent;
    main_1();
    let _e29 = v_WorldPosition;
    let _e31 = v_WorldNormal;
    let _e33 = v_Uv;
    let _e35 = v_WorldTangent;
    let _e37 = gl_Position;
    return VertexOutput(_e29, _e31, _e33, _e35, _e37);
}
