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

    let _e3 = global.Color;
    color = _e3;
    let _e5 = color;
    o_Target = _e5;
    return;
}

@fragment 
fn main(@location(0) v_Uv: vec2<f32>) -> FragmentOutput {
    v_Uv_1 = v_Uv;
    main_1();
    let _e3 = o_Target;
    return FragmentOutput(_e3);
}
