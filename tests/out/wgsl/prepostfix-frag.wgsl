fn main1() {
    var scalar_target: i32;
    var scalar: i32 = 1;
    var vec_target: vec2<u32>;
    var vec1: vec2<u32> = vec2<u32>(1u, 1u);
    var mat_target: mat4x3<f32>;
    var mat1: mat4x3<f32> = mat4x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0));

    let e3: i32 = scalar;
    scalar = (e3 + 1);
    scalar_target = e3;
    let e6: i32 = scalar;
    let e8: i32 = (e6 - 1);
    scalar = e8;
    scalar_target = e8;
    let e14: vec2<u32> = vec1;
    vec1 = (e14 - vec2<u32>(1u));
    vec_target = e14;
    let e18: vec2<u32> = vec1;
    let e21: vec2<u32> = (e18 + vec2<u32>(1u));
    vec1 = e21;
    vec_target = e21;
    let e24: f32 = f32(1);
    let e32: mat4x3<f32> = mat1;
    let e34: vec3<f32> = vec3<f32>(1.0);
    mat1 = (e32 + mat4x3<f32>(e34, e34, e34, e34));
    mat_target = e32;
    let e37: mat4x3<f32> = mat1;
    let e39: vec3<f32> = vec3<f32>(1.0);
    let e41: mat4x3<f32> = (e37 - mat4x3<f32>(e39, e39, e39, e39));
    mat1 = e41;
    mat_target = e41;
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}
