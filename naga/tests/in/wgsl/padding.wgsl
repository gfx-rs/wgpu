struct S {
    a: vec3<f32>,
}

struct Test {
    a: S,
    b: f32, // offset: 16
}

struct Test2 {
    a: array<vec3<f32>, 2>,
    b: f32, // offset: 32
}

struct Test3 {
    a: mat4x3<f32>,
    b: f32, // offset: 64
}

@group(0) @binding(0)
var<uniform> input1: Test;

@group(0) @binding(1)
var<uniform> input2: Test2;

@group(0) @binding(2)
var<uniform> input3: Test3;


@vertex
fn vertex() -> @builtin(position) vec4<f32> {
    return vec4<f32>(1.0) * input1.b * input2.b * input3.b;
}
