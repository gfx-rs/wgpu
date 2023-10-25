struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
}

var<private> v_uv_1: vec2<f32>;
var<private> o_color: vec4<f32>;

fn main_1() {
    o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

@fragment 
fn main(@location(0) v_uv: vec2<f32>) -> FragmentOutput {
    v_uv_1 = v_uv;
    main_1();
    let _e7 = o_color;
    return FragmentOutput(_e7);
}
