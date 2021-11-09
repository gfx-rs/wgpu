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

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: Camera;
[[group(2), binding(0)]]
var<uniform> global_1: Transform;
[[group(2), binding(1)]]
var<uniform> global_2: Sprite_size;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var position: vec3<f32>;

    let e10: vec2<f32> = Vertex_Uv_1;
    v_Uv = e10;
    let e11: vec3<f32> = Vertex_Position_1;
    let e12: vec2<f32> = global_2.size;
    position = (e11 * vec3<f32>(e12, 1.0));
    let e18: mat4x4<f32> = global.ViewProj;
    let e19: mat4x4<f32> = global_1.Model;
    let e21: vec3<f32> = position;
    gl_Position = ((e18 * e19) * vec4<f32>(e21, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0)]] Vertex_Position: vec3<f32>, [[location(1)]] Vertex_Normal: vec3<f32>, [[location(2)]] Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let e21: vec2<f32> = v_Uv;
    let e23: vec4<f32> = gl_Position;
    return VertexOutput(e21, e23);
}
