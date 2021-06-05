struct VertexOutput {
    [[location(0), interpolate(perspective)]] member: vec2<f32>;
    [[builtin(position)]] member1: vec4<f32>;
};

struct FragmentOutput {
    [[location(0), interpolate(perspective)]] member2: vec4<f32>;
};

var<private> a_pos: vec2<f32>;
var<private> a_uv: vec2<f32>;
var<private> v_uv: vec2<f32>;
var<private> gl_Position: vec4<f32>;
var<private> v_uv1: vec2<f32>;
var<private> o_color: vec4<f32>;

fn vert_main() {
    v_uv = a_uv;
    gl_Position = vec4<f32>((1.2000000476837158 * a_pos), 0.0, 1.0);
    return;
}

fn frag_main() {
    o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

[[stage(vertex)]]
fn vert_main1([[location(0), interpolate(perspective)]] param: vec2<f32>, [[location(1), interpolate(perspective)]] param1: vec2<f32>) -> VertexOutput {
    a_pos = param;
    a_uv = param1;
    vert_main();
    return VertexOutput(v_uv, gl_Position);
}

[[stage(fragment)]]
fn frag_main1() -> FragmentOutput {
    frag_main();
    return FragmentOutput(o_color);
}
