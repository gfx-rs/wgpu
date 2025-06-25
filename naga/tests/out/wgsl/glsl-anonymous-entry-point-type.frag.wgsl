struct FragmentOutput {
    @location(0) o_Target: vec4<f32>,
}

var<private> o_Target: vec4<f32>;

fn main_1() {
    o_Target = vec4(0f);
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e1 = o_Target;
    return FragmentOutput(_e1);
}
