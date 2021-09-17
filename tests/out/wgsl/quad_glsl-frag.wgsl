struct FragmentOutput {
    [[location(0)]] o_color: vec4<f32>;
};

var<private> v_uv1: vec2<f32>;
var<private> o_color: vec4<f32>;

fn main1() {
    o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] v_uv: vec2<f32>) -> FragmentOutput {
    v_uv1 = v_uv;
    main1();
    let e7: vec4<f32> = o_color;
    return FragmentOutput(e7);
}
