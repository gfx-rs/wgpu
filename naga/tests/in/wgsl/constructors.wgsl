struct Foo {
    a: vec4<f32>,
    b: i32,
}

const const1 = vec3<f32>(0.0);
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
const cp1 = vec2(0u);
const cp2 = mat2x2(vec2(0.), vec2(0.));
const cp3 = array(0, 1, 2, 3);

// complex composites: a vector ZeroValue or Splat as a constructor argument
const ccz0 = vec4<f32>(vec2<f32>(), 1.0, 1.0);    // ZeroValue(vec2) component
const ccz1 = vec4<f32>(vec2<f32>(1.0), 0.0, 1.0); // Splat(vec2) component
const ccz2 = vec4<f32>(vec2<f32>(1.0, 0.0), 0.0, 1.0); // vec2 component

// matrix composites: columns are vectors, so a ZeroValue/Splat column must be
// kept as a single (vector) constituent, not expanded into scalars
const ccm0 = mat2x2<f32>(vec2<f32>(), vec2<f32>(1.0, 2.0)); // ZeroValue(vec2) column
const ccm1 = mat2x2<f32>(vec2<f32>(1.0), vec2<f32>());      // Splat + ZeroValue columns

// a nested ZeroValue vector must be expanded into scalars at every level of
// nesting, not just at the top level of a vector composite
const ccn0 = vec4<f32>(vec3<f32>(vec2<f32>(), 3.0), 4.0);

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
    let zvc8: vec2<u32> = vec2();
    let zvc9: vec2<f32> = vec2();

    // constructors that infer their type from their parameters
    let cit0 = vec2(0u);
    let cit1 = mat2x2(vec2(0.), vec2(0.));
    let cit2 = array(0, 1, 2, 3);

    // complex composites: a vector ZeroValue or Splat as a constructor argument
    let ccz3 = vec4<f32>(vec2<f32>(), 1.0, 1.0);
    let ccz4 = vec4<f32>(vec2<f32>(1.0), 0.0, 1.0);
    let ccz5 = vec4<f32>(vec2<f32>(1.0, 0.0), 0.0, 1.0);

    // matrix composites: ZeroValue/Splat columns kept as vector constituents
    let ccm2 = mat2x2<f32>(vec2<f32>(), vec2<f32>(1.0, 2.0));
    let ccm3 = mat2x2<f32>(vec2<f32>(1.0), vec2<f32>());

    // a nested ZeroValue vector, expanded at every level of nesting
    let ccn1 = vec4<f32>(vec3<f32>(vec2<f32>(), 3.0), 4.0);

    // identity constructors
    let ic0 = bool(bool());
    let ic1 = i32(i32());
    let ic2 = u32(u32());
    let ic3 = f32(f32());
    let ic4 = vec2<u32>(vec2<u32>());
    let ic5 = mat2x3<f32>(mat2x3<f32>());
    let ic6 = vec2(vec2<u32>());
    let ic7 = mat2x3(mat2x3<f32>());

    // conversion constructors
    let cc00 = i32(1u);
    let cc01 = i32(1f);
    let cc02 = i32(1);
    let cc03 = i32(1.0);
    let cc04 = i32(true);
    let cc05 = u32(1i);
    let cc06 = u32(1f);
    let cc07 = u32(1);
    let cc08 = u32(1.0);
    let cc09 = u32(true);
    let cc10 = f32(1i);
    let cc11 = f32(1u);
    let cc12 = f32(1);
    let cc13 = f32(1.0);
    let cc14 = f32(true);
    let cc15 = bool(1i);
    let cc16 = bool(1u);
    let cc17 = bool(1f);
    let cc18 = bool(1);
    let cc19 = bool(1.0);
}
