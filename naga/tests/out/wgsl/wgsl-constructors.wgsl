struct Foo {
    a: vec4<f32>,
    b: i32,
}

const const1_: vec3<f32> = vec3(0f);
const const3_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0f, 1f), vec2<f32>(2f, 3f));
const const4_: array<mat2x2<f32>, 1> = array<mat2x2<f32>, 1>(mat2x2<f32>(vec2<f32>(0f, 1f), vec2<f32>(2f, 3f)));
const cz0_: bool = bool();
const cz1_: i32 = i32();
const cz2_: u32 = u32();
const cz3_: f32 = f32();
const cz4_: vec2<u32> = vec2<u32>();
const cz5_: mat2x2<f32> = mat2x2<f32>();
const cz6_: array<Foo, 3> = array<Foo, 3>();
const cz7_: Foo = Foo();
const cp1_: vec2<u32> = vec2(0u);
const ccz0_: vec4<f32> = vec4<f32>(vec2<f32>(), 1f, 1f);
const ccz1_: vec4<f32> = vec4<f32>(vec2(1f), 0f, 1f);
const ccz2_: vec4<f32> = vec4<f32>(vec2<f32>(1f, 0f), 0f, 1f);
const ccm0_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(), vec2<f32>(1f, 2f));
const ccm1_: mat2x2<f32> = mat2x2<f32>(vec2(1f), vec2<f32>());
const ccn0_: vec4<f32> = vec4<f32>(vec3<f32>(vec2<f32>(), 3f), 4f);

@compute @workgroup_size(1, 1, 1)
fn main() {
    var foo: Foo;

    foo = Foo(vec4(1f), 1i);
    let m0_ = mat2x2<f32>(vec2<f32>(1f, 0f), vec2<f32>(0f, 1f));
    let m1_ = mat4x4<f32>(vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(0f, 1f, 0f, 0f), vec4<f32>(0f, 0f, 1f, 0f), vec4<f32>(0f, 0f, 0f, 1f));
    let zvc8_ = vec2<u32>(0u, 0u);
    let zvc9_ = vec2<f32>(0f, 0f);
    let cit0_ = vec2(0u);
    let cit1_ = mat2x2<f32>(vec2(0f), vec2(0f));
    let cit2_ = array<i32, 4>(0i, 1i, 2i, 3i);
    let ccz3_ = vec4<f32>(vec2<f32>(), 1f, 1f);
    let ccz4_ = vec4<f32>(vec2(1f), 0f, 1f);
    let ccz5_ = vec4<f32>(vec2<f32>(1f, 0f), 0f, 1f);
    let ccm2_ = mat2x2<f32>(vec2<f32>(), vec2<f32>(1f, 2f));
    let ccm3_ = mat2x2<f32>(vec2(1f), vec2<f32>());
    let ccn1_ = vec4<f32>(vec3<f32>(vec2<f32>(), 3f), 4f);
    let ic4_ = vec2<u32>(0u, 0u);
    let ic5_ = mat2x3<f32>(vec3<f32>(0f, 0f, 0f), vec3<f32>(0f, 0f, 0f));
    return;
}
