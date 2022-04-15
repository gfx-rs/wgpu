struct Camera {
    ViewProj: mat4x4<f32>,
}

struct Transform {
    Model: mat4x4<f32>,
}

struct Sprite_size {
    size: vec2<f32>,
}

struct VertexOutput {
    @location(0) v_Uv: vec2<f32>,
    @builtin(position) member: vec4<f32>,
}

var<private> Vertex_Position_1: vec3<f32>;
var<private> Vertex_Normal_1: vec3<f32>;
var<private> Vertex_Uv_1: vec2<f32>;
var<private> v_Uv: vec2<f32>;
@group(0) @binding(0) 
var<uniform> global: Camera;
@group(2) @binding(0) 
var<uniform> global_1: Transform;
@group(2) @binding(1) 
var<uniform> global_2: Sprite_size;
var<private> gl_Position: vec4<f32>;

fn main_1() {
    var position: vec3<f32>;

    let _e10 = Vertex_Uv_1;
    v_Uv = _e10;
    let _e11 = Vertex_Position_1;
    let _e12 = global_2.size;
    position = (_e11 * vec3<f32>(_e12.x, _e12.y, 1.0));
    let _e20 = global.ViewProj;
    let _e21 = global_1.Model;
    let _e23 = position;
    gl_Position = ((_e20 * _e21) * vec4<f32>(_e23.x, _e23.y, _e23.z, 1.0));
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let _e21 = v_Uv;
    let _e23 = gl_Position;
    return VertexOutput(_e21, _e23);
}
