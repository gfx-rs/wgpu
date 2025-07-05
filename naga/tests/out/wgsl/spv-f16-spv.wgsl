enable f16;

struct B {
    b_1_: f16,
    b_vec2_: vec2<f16>,
    b_vec3_: vec3<f16>,
    b_vec4_: vec4<f16>,
    b_mat2_: mat2x2<f16>,
    b_mat2x3_: mat2x3<f16>,
    b_mat2x4_: mat2x4<f16>,
    b_mat3x2_: mat3x2<f16>,
    b_mat3_: mat3x3<f16>,
    b_mat3x4_: mat3x4<f16>,
    b_mat4x2_: mat4x2<f16>,
    b_mat4x3_: mat4x3<f16>,
    b_mat4_: mat4x4<f16>,
}

struct A {
    a_1_: f16,
    a_vec2_: vec2<f16>,
    a_vec3_: vec3<f16>,
    a_vec4_: vec4<f16>,
}

@group(0) @binding(1) 
var<storage, read_write> unnamed: B;
@group(0) @binding(0) 
var<uniform> unnamed_1: A;

fn main_1() {
    let _e3 = unnamed_1.a_1_;
    unnamed.b_1_ = _e3;
    let _e6 = unnamed_1.a_vec2_;
    unnamed.b_vec2_ = _e6;
    let _e9 = unnamed_1.a_vec3_;
    unnamed.b_vec3_ = _e9;
    let _e12 = unnamed_1.a_vec4_;
    unnamed.b_vec4_ = _e12;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
}
