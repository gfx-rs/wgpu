[[block]]
struct ColorMaterial_color {
    Color: vec4<f32>;
};

struct FragmentOutput {
    [[location(0), interpolate(perspective)]] member: vec4<f32>;
};

var<private> gen_entry_v_Uv: vec2<f32>;
var<private> gen_entry_o_Target: vec4<f32>;
[[group(1), binding(0)]]
var<uniform> global: ColorMaterial_color;

fn main() {
    var color: vec4<f32>;

    let _e4: vec4<f32> = global.Color;
    color = _e4;
    let _e6: vec4<f32> = color;
    gen_entry_o_Target = _e6;
    return;
}

[[stage(fragment)]]
fn main1() -> FragmentOutput {
    main();
    let _e1: vec4<f32> = gen_entry_o_Target;
    return FragmentOutput(_e1);
}
