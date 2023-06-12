struct Mat4x3_ {
    mx: vec4<f32>,
    my: vec4<f32>,
    mz: vec4<f32>,
}

struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
}

var<private> o_color: vec4<f32>;

fn Fma(d: ptr<function, Mat4x3_>, m: Mat4x3_, s: f32) {
    var m_1: Mat4x3_;
    var s_1: f32;

    m_1 = m;
    s_1 = s;
    let _e6 = (*d);
    let _e8 = m_1;
    let _e10 = s_1;
    (*d).mx = (_e6.mx + (_e8.mx * _e10));
    let _e14 = (*d);
    let _e16 = m_1;
    let _e18 = s_1;
    (*d).my = (_e14.my + (_e16.my * _e18));
    let _e22 = (*d);
    let _e24 = m_1;
    let _e26 = s_1;
    (*d).mz = (_e22.mz + (_e24.mz * _e26));
    return;
}

fn main_1() {
    let _e1 = o_color;
    let _e4 = vec4<f32>(1.0);
    o_color.x = _e4.x;
    o_color.y = _e4.y;
    o_color.z = _e4.z;
    o_color.w = _e4.w;
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e3 = o_color;
    return FragmentOutput(_e3);
}
