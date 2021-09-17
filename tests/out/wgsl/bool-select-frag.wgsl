struct FragmentOutput {
    [[location(0)]] o_color: vec4<f32>;
};

var<private> o_color: vec4<f32>;

fn TevPerCompGT(a: f32, b: f32) -> f32 {
    var a1: f32;
    var b1: f32;

    a1 = a;
    b1 = b;
    let e5: f32 = a1;
    let e6: f32 = b1;
    return select(0.0, 1.0, (e5 > e6));
}

fn TevPerCompGT1(a2: vec3<f32>, b2: vec3<f32>) -> vec3<f32> {
    var a3: vec3<f32>;
    var b3: vec3<f32>;

    a3 = a2;
    b3 = b2;
    let e7: vec3<f32> = a3;
    let e8: vec3<f32> = b3;
    return select(vec3<f32>(0.0), vec3<f32>(1.0), (e7 > e8));
}

fn main1() {
    let e1: vec4<f32> = o_color;
    let e11: vec3<f32> = TevPerCompGT1(vec3<f32>(3.0), vec3<f32>(5.0));
    o_color.x = e11.x;
    o_color.y = e11.y;
    o_color.z = e11.z;
    let e23: f32 = TevPerCompGT(3.0, 5.0);
    o_color.w = e23;
    return;
}

[[stage(fragment)]]
fn main() -> FragmentOutput {
    main1();
    let e3: vec4<f32> = o_color;
    return FragmentOutput(e3);
}
