struct Mat4x3_ {
    mx: vec4<f32>;
    my: vec4<f32>;
    mz: vec4<f32>;
};

var<private> o_color: vec4<f32>;

fn Fma(d: ptr<function, Mat4x3_>, m: Mat4x3_, s: f32) {
    var m_1: Mat4x3_;
    var s_1: f32;

    m_1 = m;
    s_1 = s;
    let e6: Mat4x3_ = (*d);
    let e8: Mat4x3_ = m_1;
    let e10: f32 = s_1;
    (*d).mx = (e6.mx + (e8.mx * e10));
    let e14: Mat4x3_ = (*d);
    let e16: Mat4x3_ = m_1;
    let e18: f32 = s_1;
    (*d).my = (e14.my + (e16.my * e18));
    let e22: Mat4x3_ = (*d);
    let e24: Mat4x3_ = m_1;
    let e26: f32 = s_1;
    (*d).mz = (e22.mz + (e24.mz * e26));
    return;
}

fn main_1() {
    let e1: vec4<f32> = o_color;
    let e4: vec4<f32> = vec4<f32>(1.0);
    o_color.x = e4.x;
    o_color.y = e4.y;
    o_color.z = e4.z;
    o_color.w = e4.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main_1();
    return;
}
