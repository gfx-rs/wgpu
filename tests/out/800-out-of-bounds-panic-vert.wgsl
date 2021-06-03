[[block]]
struct Globals {
    view_matrix: mat4x4<f32>;
};

[[block]]
struct VertexPushConstants {
    world_matrix: mat4x4<f32>;
};

struct VertexOutput {
    [[location(0), interpolate(perspective)]] member: vec4<f32>;
    [[builtin(position)]] member1: vec4<f32>;
};

[[group(0), binding(0)]]
var<uniform> global: Globals;
var<push_constant> global1: VertexPushConstants;
var<private> position: vec2<f32>;
var<private> color: vec4<f32>;
var<private> frag_color: vec4<f32>;
var<private> gl_Position: vec4<f32>;

fn main() {
    frag_color = color;
    gl_Position = ((global.view_matrix * global1.world_matrix) * vec4<f32>(position, 0.0, 1.0));
    return;
}

[[stage(vertex)]]
fn main1([[location(0), interpolate(perspective)]] param: vec2<f32>, [[location(1), interpolate(perspective)]] param1: vec4<f32>) -> VertexOutput {
    position = param;
    color = param1;
    main();
    return VertexOutput(frag_color, gl_Position);
}
