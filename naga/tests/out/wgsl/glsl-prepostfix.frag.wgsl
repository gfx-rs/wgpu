fn main_1() {
    var scalar_target: i32;
    var scalar: i32 = 1i;
    var vec_target: vec2<u32>;
    var vec: vec2<u32> = vec2(1u);
    var mat_target: mat4x3<f32>;
    var mat: mat4x3<f32> = mat4x3<f32>(vec3<f32>(1f, 0f, 0f), vec3<f32>(0f, 1f, 0f), vec3<f32>(0f, 0f, 1f), vec3<f32>(0f, 0f, 0f));

    let _e3 = scalar;
    scalar = (_e3 + 1i);
    scalar_target = _e3;
    let _e6 = scalar;
    let _e8 = (_e6 - 1i);
    scalar = _e8;
    scalar_target = _e8;
    let _e13 = vec;
    vec = (_e13 - vec2(1u));
    vec_target = _e13;
    let _e17 = vec;
    let _e20 = (_e17 + vec2(1u));
    vec = _e20;
    vec_target = _e20;
    let _e30 = mat;
    let _e32 = vec3(1f);
    mat = (_e30 + mat4x3<f32>(_e32, _e32, _e32, _e32));
    mat_target = _e30;
    let _e35 = mat;
    let _e37 = vec3(1f);
    let _e39 = (_e35 - mat4x3<f32>(_e37, _e37, _e37, _e37));
    mat = _e39;
    mat_target = _e39;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
