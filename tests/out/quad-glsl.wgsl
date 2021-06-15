struct VertexOutput {
    [[location(0), interpolate(perspective)]] member: vec2<f32>;
    [[builtin(position)]] member1: vec4<f32>;
};

struct FragmentOutput {
    [[location(0), interpolate(perspective)]] member2: vec4<f32>;
};

var<private> gen_entry_a_pos: vec2<f32>;
var<private> gen_entry_a_uv: vec2<f32>;
var<private> gen_entry_v_uv: vec2<f32>;
var<private> gl_Position: vec4<f32>;
var<private> gen_entry_v_uv1: vec2<f32>;
var<private> gen_entry_o_color: vec4<f32>;

fn vert_main() {
    let _e2: vec2<f32> = gen_entry_a_pos;
    let _e4: vec2<f32> = gen_entry_a_uv;
    gen_entry_v_uv = _e4;
    gl_Position = vec4<f32>((1.2000000476837158 * _e2), 0.0, 1.0);
    return;
}

fn frag_main() {
    gen_entry_o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

[[stage(vertex)]]
fn vert_main1([[location(0), interpolate(perspective)]] a_pos: vec2<f32>, [[location(1), interpolate(perspective)]] a_uv: vec2<f32>) -> VertexOutput {
    gen_entry_a_pos = a_pos;
    gen_entry_a_uv = a_uv;
    vert_main();
    let _e5: vec2<f32> = gen_entry_v_uv;
    let _e7: vec4<f32> = gl_Position;
    return VertexOutput(_e5, _e7);
}

[[stage(fragment)]]
fn frag_main1() -> FragmentOutput {
    frag_main();
    let _e1: vec4<f32> = gen_entry_o_color;
    return FragmentOutput(_e1);
}
