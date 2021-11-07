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

    let e12: mat4x4<f32> = global_1.Model;
    let e13: vec3<f32> = Vertex_Position_1;
    world_position = (e12 * vec4<f32>(e13, 1.0));
    let e18: vec4<f32> = world_position;
    v_WorldPosition = e18.xyz;
    let e20: mat4x4<f32> = global_1.Model;
    let e30: vec3<f32> = Vertex_Normal_1;
    v_WorldNormal = (mat3x3<f32>(e20[0].xyz, e20[1].xyz, e20[2].xyz) * e30);
    let e32: vec2<f32> = Vertex_Uv_1;
    v_Uv = e32;
    let e33: mat4x4<f32> = global_1.Model;
    let e43: vec4<f32> = Vertex_Tangent_1;
    let e46: vec4<f32> = Vertex_Tangent_1;
    v_WorldTangent = vec4<f32>((mat3x3<f32>(e33[0].xyz, e33[1].xyz, e33[2].xyz) * e43.xyz), e46.w);
    let e50: mat4x4<f32> = global.ViewProj;
    let e51: vec4<f32> = world_position;
    gl_Position = (e50 * e51);
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(1)]] Vertex_Normal: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>, [[location(3)]] Vertex_Tangent: vec4<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    Vertex_Tangent_1 = Vertex_Tangent;
    main_1();
    let e29: vec3<f32> = v_WorldPosition;
    let e31: vec3<f32> = v_WorldNormal;
    let e33: vec2<f32> = v_Uv;
    let e35: vec4<f32> = v_WorldTangent;
    let e37: vec4<f32> = gl_Position;
    return VertexOutput(e29, e31, e33, e35, e37);
}
