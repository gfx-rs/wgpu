[[block]]
struct ColorMaterial_color {
    Color: vec4<f32>;
};

struct FragmentOutput {
    [[location(0), interpolate(perspective)]] member: vec4<f32>;
};

var<private> v_Uv: vec2<f32>;
var<private> o_Target: vec4<f32>;
[[group(1), binding(0)]]
var<uniform> global: ColorMaterial_color;

fn main() {
    var color: vec4<f32>;

    color = global.Color;
    o_Target = color;
    return;
}

[[stage(fragment)]]
fn main1() -> FragmentOutput {
    main();
    return FragmentOutput(o_Target);
}
