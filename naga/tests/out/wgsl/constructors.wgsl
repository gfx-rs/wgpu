struct Foo {
    a: vec4<f32>,
    b: i32,
}

const const2_: vec3<f32> = vec3<f32>(0.0, 1.0, 2.0);
const const3_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.0, 1.0), vec2<f32>(2.0, 3.0));
const const4_: array<mat2x2<f32>, 1> = array<mat2x2<f32>, 1>(mat2x2<f32>(vec2<f32>(0.0, 1.0), vec2<f32>(2.0, 3.0)));
const cz0_: bool = bool();
const cz1_: i32 = i32();
const cz2_: u32 = u32();
const cz3_: f32 = f32();
const cz4_: vec2<u32> = vec2<u32>();
const cz5_: mat2x2<f32> = mat2x2<f32>();
const cz6_: array<Foo, 3> = array<Foo, 3>();
const cz7_: Foo = Foo();
const cp3_: array<i32, 4> = array<i32, 4>(0, 1, 2, 3);

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var foo: Foo;

    foo = Foo(vec4(1.0), 1);
    let m0_ = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let m1_ = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let cit0_ = vec2(0u);
    let cit1_ = mat2x2<f32>(vec2(0.0), vec2(0.0));
    let cit2_ = array<i32, 4>(0, 1, 2, 3);
    let ic0_ = bool(bool());
    let ic4_ = vec2<u32>(0u, 0u);
    let ic5_ = mat2x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
}
