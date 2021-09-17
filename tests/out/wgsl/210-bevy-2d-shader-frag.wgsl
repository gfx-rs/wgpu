[[block]]
struct ColorMaterial_color {
    Color: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] o_Target: vec4<f32>;
};

var<private> v_Uv1: vec2<f32>;
var<private> o_Target: vec4<f32>;
[[group(1), binding(0)]]
var<uniform> global: ColorMaterial_color;

fn main1() {
    var color: vec4<f32>;

    let e4: vec4<f32> = global.Color;
    color = e4;
    let e6: vec4<f32> = color;
    o_Target = e6;
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] v_Uv: vec2<f32>) -> FragmentOutput {
    v_Uv1 = v_Uv;
    main1();
    let e9: vec4<f32> = o_Target;
    return FragmentOutput(e9);
}
