struct S {
    a: vec3<f32>,
}

struct Test {
    a: S,
    b: f32,
}

struct Test2_ {
    a: array<vec3<f32>, 2>,
    b: f32,
}

struct Test3_ {
    a: mat4x3<f32>,
    b: f32,
}

@group(0) @binding(0) 
var<uniform> input1_: Test;
@group(0) @binding(1) 
var<uniform> input2_: Test2_;
@group(0) @binding(2) 
var<uniform> input3_: Test3_;

@vertex 
fn vertex() -> @builtin(position) vec4<f32> {
    let _e4 = input1_.b;
    let _e8 = input2_.b;
    let _e12 = input3_.b;
    return (((vec4(1.0) * _e4) * _e8) * _e12);
}
