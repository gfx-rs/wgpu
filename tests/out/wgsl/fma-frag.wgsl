struct Mat4x3_ {
    mx: vec4<f32>;
    my: vec4<f32>;
    mz: vec4<f32>;
};

var<private> o_color: vec4<f32>;

fn Fma(d: ptr<function, Mat4x3_>, m: Mat4x3_, s: f32) {
    var m1: Mat4x3_;
    var s1: f32;

    m1 = m;
    s1 = s;
    let _e6: Mat4x3_ = (*d);
    let _e8: Mat4x3_ = m1;
    let _e10: f32 = s1;
    (*d).mx = (_e6.mx + (_e8.mx * _e10));
    let _e14: Mat4x3_ = (*d);
    let _e16: Mat4x3_ = m1;
    let _e18: f32 = s1;
    (*d).my = (_e14.my + (_e16.my * _e18));
    let _e22: Mat4x3_ = (*d);
    let _e24: Mat4x3_ = m1;
    let _e26: f32 = s1;
    (*d).mz = (_e22.mz + (_e24.mz * _e26));
    return;
}

fn main1() {
    let _e1: vec4<f32> = o_color;
    let _e4: vec4<f32> = vec4<f32>(1.0);
    o_color.x = _e4.x;
    o_color.y = _e4.y;
    o_color.z = _e4.z;
    o_color.w = _e4.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}
