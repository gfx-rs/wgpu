struct Foo {
    a: vec4<f32>,
    b: i32,
}

// const const1 = vec3<f32>(0.0); // TODO: this is now a splat and we need to const eval it
const const2 = vec3(0.0, 1.0, 2.0);
const const3 = mat2x2<f32>(0.0, 1.0, 2.0, 3.0);
const const4 = array<mat2x2<f32>, 1>(mat2x2<f32>(0.0, 1.0, 2.0, 3.0));

// zero value constructors
const cz0 = bool();
const cz1 = i32();
const cz2 = u32();
const cz3 = f32();
const cz4 = vec2<u32>();
const cz5 = mat2x2<f32>();
const cz6 = array<Foo, 3>();
const cz7 = Foo();

// constructors that infer their type from their parameters
// TODO: these also contain splats
// const cp1 = vec2(0u);
// const cp2 = mat2x2(vec2(0.), vec2(0.));
const cp3 = array(0, 1, 2, 3);

@compute @workgroup_size(1)
fn main() {
    var foo: Foo;
    foo = Foo(vec4<f32>(1.0), 1);

    let m0 = mat2x2<f32>(
        1.0, 0.0,
        0.0, 1.0,
    );
    let m1 = mat4x4<f32>(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    // zero value constructors
    let zvc0 = bool();
    let zvc1 = i32();
    let zvc2 = u32();
    let zvc3 = f32();
    let zvc4 = vec2<u32>();
    let zvc5 = mat2x2<f32>();
    let zvc6 = array<Foo, 3>();
    let zvc7 = Foo();

    // constructors that infer their type from their parameters
    let cit0 = vec2(0u);
    let cit1 = mat2x2(vec2(0.), vec2(0.));
    let cit2 = array(0, 1, 2, 3);

    // identity constructors
    let ic0 = bool(bool());
    let ic1 = i32(i32());
    let ic2 = u32(u32());
    let ic3 = f32(f32());
    let ic4 = vec2<u32>(vec2<u32>());
    let ic5 = mat2x3<f32>(mat2x3<f32>());
    let ic6 = vec2(vec2<u32>());
    let ic7 = mat2x3(mat2x3<f32>());
}
