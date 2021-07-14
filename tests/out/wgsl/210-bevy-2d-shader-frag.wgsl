[[block]]
struct ColorMaterial_color {
    Color: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] o_Target: vec4<f32>;
};

var<private> v_Uv: vec2<f32>;
var<private> o_Target: vec4<f32>;
[[group(1), binding(0)]]
var<uniform> global: ColorMaterial_color;

fn main1() {
    var color: vec4<f32>;

    let _e4: vec4<f32> = global.Color;
    color = _e4;
    let _e6: vec4<f32> = color;
    o_Target = _e6;
    return;
}

[[stage(fragment)]]
fn main() -> FragmentOutput {
    main1();
    let _e1: vec4<f32> = o_Target;
    return FragmentOutput(_e1);
}
