struct FragmentOutput {
    [[location(0)]] o_color: vec4<f32>;
};

var<private> v_uv: vec2<f32>;
var<private> o_color: vec4<f32>;

fn main1() {
    o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

[[stage(fragment)]]
fn main() -> FragmentOutput {
    main1();
    let _e1: vec4<f32> = o_color;
    return FragmentOutput(_e1);
}
