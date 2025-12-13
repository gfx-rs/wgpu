enable f16;

struct A {
    a_1_: f16,
    a_vec2_: vec2<f16>,
    a_vec3_: vec3<f16>,
    a_vec4_: vec4<f16>,
}

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

@group(0) @binding(0) 
var<uniform> global: A;
@group(0) @binding(1) 
var<storage, read_write> global_1: B;

fn main_1() {
    let _e16 = global.a_1_;
    global_1.b_1_ = _e16;
    let _e17 = global.a_vec2_;
    global_1.b_vec2_ = _e17;
    let _e18 = global.a_vec3_;
    global_1.b_vec3_ = _e18;
    let _e19 = global.a_vec4_;
    global_1.b_vec4_ = _e19;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
    return;
}
