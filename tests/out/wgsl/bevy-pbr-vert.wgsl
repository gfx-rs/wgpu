[[block]]
struct CameraViewProj {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct Transform {
    Model: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0)]] v_WorldPosition: vec3<f32>;
    [[location(1)]] v_WorldNormal: vec3<f32>;
    [[location(2)]] v_Uv: vec2<f32>;
    [[location(3)]] v_WorldTangent: vec4<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> Vertex_Tangent_1: vec4<f32>;
var<private> v_WorldPosition: vec3<f32>;
var<private> v_WorldNormal: vec3<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: CameraViewProj;
var<private> v_WorldTangent: vec4<f32>;
[[group(2), binding(0)]]
var<uniform> global_1: Transform;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var world_position: vec4<f32>;

    let _e12: mat4x4<f32> = global_1.Model;
    let _e13: vec3<f32> = Vertex_Position_1;
    world_position = (_e12 * vec4<f32>(_e13, 1.0));
    let _e18: vec4<f32> = world_position;
    v_WorldPosition = _e18.xyz;
    let _e20: mat4x4<f32> = global_1.Model;
    let _e30: vec3<f32> = Vertex_Normal_1;
    v_WorldNormal = (mat3x3<f32>(_e20[0].xyz, _e20[1].xyz, _e20[2].xyz) * _e30);
    let _e32: vec2<f32> = Vertex_Uv_1;
    v_Uv = _e32;
    let _e33: mat4x4<f32> = global_1.Model;
    let _e43: vec4<f32> = Vertex_Tangent_1;
    let _e46: vec4<f32> = Vertex_Tangent_1;
    v_WorldTangent = vec4<f32>((mat3x3<f32>(_e33[0].xyz, _e33[1].xyz, _e33[2].xyz) * _e43.xyz), _e46.w);
    let _e50: mat4x4<f32> = global.ViewProj;
    let _e51: vec4<f32> = world_position;
    gl_Position = (_e50 * _e51);
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(1)]] Vertex_Normal: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>, [[location(3)]] Vertex_Tangent: vec4<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    Vertex_Tangent_1 = Vertex_Tangent;
    main_1();
    let _e29: vec3<f32> = v_WorldPosition;
    let _e31: vec3<f32> = v_WorldNormal;
    let _e33: vec2<f32> = v_Uv;
    let _e35: vec4<f32> = v_WorldTangent;
    let _e37: vec4<f32> = gl_Position;
    return VertexOutput(_e29, _e31, _e33, _e35, _e37);
}
