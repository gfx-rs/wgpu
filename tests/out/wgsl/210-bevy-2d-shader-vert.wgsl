[[block]]
struct Camera {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct Transform {
    Model: mat4x4<f32>;
};

[[block]]
struct Sprite_size {
    size: vec2<f32>;
};

struct VertexOutput {
    [[location(0)]] v_Uv: vec2<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

var<private> Vertex_Position1: vec3<f32>;
var<private> Vertex_Normal: vec3<f32>;
var<private> Vertex_Uv1: vec2<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: Camera;
[[group(2), binding(0)]]
var<uniform> global1: Transform;
[[group(2), binding(1)]]
var<uniform> global2: Sprite_size;
var<private> gl_Position: vec4<f32>;

fn main1() {
    var position: vec3<f32>;

    let _e10: vec2<f32> = Vertex_Uv1;
    v_Uv = _e10;
    let _e11: vec3<f32> = Vertex_Position1;
    let _e12: vec2<f32> = global2.size;
    position = (_e11 * vec3<f32>(_e12, 1.0));
    let _e18: mat4x4<f32> = global.ViewProj;
    let _e19: mat4x4<f32> = global1.Model;
    let _e21: vec3<f32> = position;
    gl_Position = ((_e18 * _e19) * vec4<f32>(_e21, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position1 = Vertex_Position;
    Vertex_Uv1 = Vertex_Uv;
    main1();
    let _e5: vec2<f32> = v_Uv;
    let _e7: vec4<f32> = gl_Position;
    return VertexOutput(_e5, _e7);
}
