[[block]]
struct CameraViewProj {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct Transform {
    Model: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0), interpolate(perspective)]] member: vec3<f32>;
    [[location(1), interpolate(perspective)]] member1: vec3<f32>;
    [[location(2), interpolate(perspective)]] member2: vec2<f32>;
    [[location(3), interpolate(perspective)]] member3: vec4<f32>;
    [[builtin(position)]] member4: vec4<f32>;
};

var<private> gen_entry_Vertex_Position: vec3<f32>;
var<private> gen_entry_Vertex_Normal: vec3<f32>;
var<private> gen_entry_Vertex_Uv: vec2<f32>;
var<private> gen_entry_Vertex_Tangent: vec4<f32>;
var<private> gen_entry_v_WorldPosition: vec3<f32>;
var<private> gen_entry_v_WorldNormal: vec3<f32>;
var<private> gen_entry_v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: CameraViewProj;
var<private> gen_entry_v_WorldTangent: vec4<f32>;
[[group(2), binding(0)]]
var<uniform> global1: Transform;
var<private> gl_Position: vec4<f32>;

fn main() {
    var world_position: vec4<f32>;

    let _e12: mat4x4<f32> = global1.Model;
    let _e13: vec3<f32> = gen_entry_Vertex_Position;
    world_position = (_e12 * vec4<f32>(_e13, 1.0));
    let _e18: vec4<f32> = world_position;
    gen_entry_v_WorldPosition = _e18.xyz;
    let _e20: mat4x4<f32> = global1.Model;
    let _e28: vec3<f32> = gen_entry_Vertex_Normal;
    gen_entry_v_WorldNormal = (mat3x3<f32>(_e20[0].xyz, _e20[1].xyz, _e20[2].xyz) * _e28);
    let _e30: vec2<f32> = gen_entry_Vertex_Uv;
    gen_entry_v_Uv = _e30;
    let _e31: mat4x4<f32> = global1.Model;
    let _e39: vec4<f32> = gen_entry_Vertex_Tangent;
    let _e42: vec4<f32> = gen_entry_Vertex_Tangent;
    gen_entry_v_WorldTangent = vec4<f32>((mat3x3<f32>(_e31[0].xyz, _e31[1].xyz, _e31[2].xyz) * _e39.xyz), _e42.w);
    let _e46: mat4x4<f32> = global.ViewProj;
    let _e47: vec4<f32> = world_position;
    gl_Position = (_e46 * _e47);
    return;
}

[[stage(vertex)]]
fn main1([[location(0), interpolate(perspective)]] Vertex_Position: vec3<f32>, [[location(1), interpolate(perspective)]] Vertex_Normal: vec3<f32>, [[location(2), interpolate(perspective)]] Vertex_Uv: vec2<f32>, [[location(3), interpolate(perspective)]] Vertex_Tangent: vec4<f32>) -> VertexOutput {
    gen_entry_Vertex_Position = Vertex_Position;
    gen_entry_Vertex_Normal = Vertex_Normal;
    gen_entry_Vertex_Uv = Vertex_Uv;
    gen_entry_Vertex_Tangent = Vertex_Tangent;
    main();
    let _e9: vec3<f32> = gen_entry_v_WorldPosition;
    let _e11: vec3<f32> = gen_entry_v_WorldNormal;
    let _e13: vec2<f32> = gen_entry_v_Uv;
    let _e15: vec4<f32> = gen_entry_v_WorldTangent;
    let _e17: vec4<f32> = gl_Position;
    return VertexOutput(_e9, _e11, _e13, _e15, _e17);
}
