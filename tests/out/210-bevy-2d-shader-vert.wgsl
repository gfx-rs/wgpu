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
    [[location(0), interpolate(perspective)]] member: vec2<f32>;
    [[builtin(position)]] member1: vec4<f32>;
};

var<private> Vertex_Position: vec3<f32>;
var<private> Vertex_Normal: vec3<f32>;
var<private> Vertex_Uv: vec2<f32>;
var<private> v_Uv: vec2<f32>;
[[group(0), binding(0)]]
var<uniform> global: Camera;
[[group(2), binding(0)]]
var<uniform> global1: Transform;
[[group(2), binding(1)]]
var<uniform> global2: Sprite_size;
var<private> gl_Position: vec4<f32>;

fn main() {
    var position: vec3<f32>;

    v_Uv = Vertex_Uv;
    position = (Vertex_Position * vec3<f32>(global2.size, 1.0));
    gl_Position = ((global.ViewProj * global1.Model) * vec4<f32>(position, 1.0));
    return;
}

[[stage(vertex)]]
fn main1([[location(0), interpolate(perspective)]] param: vec3<f32>, [[location(2), interpolate(perspective)]] param1: vec2<f32>) -> VertexOutput {
    Vertex_Position = param;
    Vertex_Uv = param1;
    main();
    return VertexOutput(v_Uv, gl_Position);
}
