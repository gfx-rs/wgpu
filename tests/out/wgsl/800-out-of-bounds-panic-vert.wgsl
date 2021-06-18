[[block]]
struct Globals {
    view_matrix: mat4x4<f32>;
};

[[block]]
struct VertexPushConstants {
    world_matrix: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0), interpolate(perspective)]] frag_color: vec4<f32>;
    [[builtin(position)]] member: vec4<f32>;
};

[[group(0), binding(0)]]
var<uniform> global: Globals;
var<push_constant> global1: VertexPushConstants;
var<private> position1: vec2<f32>;
var<private> color1: vec4<f32>;
var<private> frag_color: vec4<f32>;
var<private> gl_Position: vec4<f32>;

fn main1() {
    let _e7: vec4<f32> = color1;
    frag_color = _e7;
    let _e9: mat4x4<f32> = global.view_matrix;
    let _e10: mat4x4<f32> = global1.world_matrix;
    let _e12: vec2<f32> = position1;
    gl_Position = ((_e9 * _e10) * vec4<f32>(_e12, 0.0, 1.0));
    return;
}

[[stage(vertex)]]
fn main([[location(0), interpolate(perspective)]] position: vec2<f32>, [[location(1), interpolate(perspective)]] color: vec4<f32>) -> VertexOutput {
    position1 = position;
    color1 = color;
    main1();
    let _e5: vec4<f32> = frag_color;
    let _e7: vec4<f32> = gl_Position;
    return VertexOutput(_e5, _e7);
}
