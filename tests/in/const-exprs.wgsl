@group(0) @binding(0) var<storage, read_write> out: vec4<i32>;
@group(0) @binding(1) var<storage, read_write> out2: i32;

const TWO: u32 = 2u;
const THREE: i32 = 3i;

@compute @workgroup_size(TWO, THREE, TWO - 1u)
fn main() {
   swizzle_of_compose();
   index_of_compose();
   compose_three_deep();
   non_constant_initializers();
   splat_of_constant();
   compose_of_constant();
}

// Swizzle the value of nested Compose expressions.
fn swizzle_of_compose() {
   out = vec4(vec2(1, 2), vec2(3, 4)).wzyx; // should assign vec4(4, 3, 2, 1);
}

// Index the value of nested Compose expressions.
fn index_of_compose() {
   out2 += vec4(vec2(1, 2), vec2(3, 4))[1]; // should assign 2
}

// Index the value of Compose expressions nested three deep
fn compose_three_deep() {
   out2 += vec4(vec3(vec2(6, 7), 8), 9)[0]; // should assign 6
}

// While WGSL allows local variables to be declared anywhere in the function,
// Naga treats them all as appearing at the top of the function. To ensure that
// WGSL initializer expressions are evaluated at the right time, in the general
// case they need to be turned into Naga `Store` statements executed at the
// point of the WGSL declaration.
//
// When a variable's initializer is a constant expression, however, it can be
// evaluated at any time. The WGSL front end thus renders locals with
// initializers that are constants as Naga locals with initializers. This test
// checks that Naga local variable initializers are only used when safe.
fn non_constant_initializers() {
   var w = 10 + 20;
   var x = w;
   var y = x;
   var z = 30 + 40;

   out += vec4(w, x, y, z);
}

// Constant evaluation should be able to see through constants to
// their values.
const FOUR: i32 = 4;

const FOUR_ALIAS: i32 = FOUR;

const TEST_CONSTANT_ADDITION: i32 = FOUR + FOUR;
const TEST_CONSTANT_ALIAS_ADDITION: i32 = FOUR_ALIAS + FOUR_ALIAS;

fn splat_of_constant() {
    out = -vec4(FOUR);
}

fn compose_of_constant() {
    out = -vec4(FOUR, FOUR, FOUR, FOUR);
}

const PI: f32 = 3.141;
const phi_sun: f32 = PI * 2.0;

const DIV: vec4f = vec4(4.0 / 9.0, 0.0, 0.0, 0.0);
