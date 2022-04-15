struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
}

var<private> o_color: vec4<f32>;

fn TevPerCompGT(a: f32, b: f32) -> f32 {
    var a_1: f32;
    var b_1: f32;

    a_1 = a;
    b_1 = b;
    let _e5 = a_1;
    let _e6 = b_1;
    return select(0.0, 1.0, (_e5 > _e6));
}

fn TevPerCompGT_1(a_2: vec3<f32>, b_2: vec3<f32>) -> vec3<f32> {
    var a_3: vec3<f32>;
    var b_3: vec3<f32>;

    a_3 = a_2;
    b_3 = b_2;
    let _e7 = a_3;
    let _e8 = b_3;
    return select(vec3<f32>(0.0), vec3<f32>(1.0), (_e7 > _e8));
}

fn main_1() {
    let _e1 = o_color;
    let _e11 = TevPerCompGT_1(vec3<f32>(3.0), vec3<f32>(5.0));
    o_color.x = _e11.x;
    o_color.y = _e11.y;
    o_color.z = _e11.z;
    let _e23 = TevPerCompGT(3.0, 5.0);
    o_color.w = _e23;
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e3 = o_color;
    return FragmentOutput(_e3);
}
