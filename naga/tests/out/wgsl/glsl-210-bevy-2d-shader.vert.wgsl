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
    @builtin(position) gl_Position: vec4<f32>,
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

    let _e9 = Vertex_Uv_1;
    v_Uv = _e9;
    let _e10 = Vertex_Position_1;
    let _e11 = global_2.size;
    position = (_e10 * vec3<f32>(_e11.x, _e11.y, 1f));
    let _e19 = global.ViewProj;
    let _e20 = global_1.Model;
    let _e22 = position;
    gl_Position = ((_e19 * _e20) * vec4<f32>(_e22.x, _e22.y, _e22.z, 1f));
    return;
}

@vertex 
fn main(@location(0) Vertex_Position: vec3<f32>, @location(1) Vertex_Normal: vec3<f32>, @location(2) Vertex_Uv: vec2<f32>) -> VertexOutput {
    Vertex_Position_1 = Vertex_Position;
    Vertex_Normal_1 = Vertex_Normal;
    Vertex_Uv_1 = Vertex_Uv;
    main_1();
    let _e7 = v_Uv;
    let _e9 = gl_Position;
    return VertexOutput(_e7, _e9);
}
