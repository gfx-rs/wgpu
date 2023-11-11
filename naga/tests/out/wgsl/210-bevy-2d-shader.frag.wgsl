struct ColorMaterial_color {
    Color: vec4<f32>,
}

struct FragmentOutput {
    @location(0) o_Target: vec4<f32>,
}

var<private> v_Uv_1: vec2<f32>;
var<private> o_Target: vec4<f32>;
@group(1) @binding(0) 
var<uniform> global: ColorMaterial_color;

fn main_1() {
    var color: vec4<f32>;

    let _e4 = global.Color;
    color = _e4;
    let _e6 = color;
    o_Target = _e6;
    return;
}

@fragment 
fn main(@location(0) v_Uv: vec2<f32>) -> FragmentOutput {
    v_Uv_1 = v_Uv;
    main_1();
    let _e9 = o_Target;
    return FragmentOutput(_e9);
}
