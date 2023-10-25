fn main_1() {
    var scalar_target: i32;
    var scalar: i32 = 1;
    var vec_target: vec2<u32>;
    var vec: vec2<u32> = vec2(1u);
    var mat_target: mat4x3<f32>;
    var mat: mat4x3<f32> = mat4x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0));

    let _e3 = scalar;
    scalar = (_e3 + 1);
    scalar_target = _e3;
    let _e6 = scalar;
    let _e8 = (_e6 - 1);
    scalar = _e8;
    scalar_target = _e8;
    let _e14 = vec;
    vec = (_e14 - vec2(1u));
    vec_target = _e14;
    let _e18 = vec;
    let _e21 = (_e18 + vec2(1u));
    vec = _e21;
    vec_target = _e21;
    let _e32 = mat;
    let _e34 = vec3(1.0);
    mat = (_e32 + mat4x3<f32>(_e34, _e34, _e34, _e34));
    mat_target = _e32;
    let _e37 = mat;
    let _e39 = vec3(1.0);
    let _e41 = (_e37 - mat4x3<f32>(_e39, _e39, _e39, _e39));
    mat = _e41;
    mat_target = _e41;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
