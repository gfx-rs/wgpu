enable dual_source_blending;

struct FragmentOutput {
    @location(0) @blend_src(0) member: vec4<f32>,
    @location(0) @blend_src(1) member_1: vec4<f32>,
}

var<private> output0_: vec4<f32>;
var<private> output1_: vec4<f32>;

fn main_1() {
    output0_ = vec4<f32>(1f, 0f, 1f, 0f);
    output1_ = vec4<f32>(0f, 1f, 0f, 1f);
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e2 = output0_;
    let _e3 = output1_;
    return FragmentOutput(_e2, _e3);
}
