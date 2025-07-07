const TWO: u32 = 2u;
const THREE: i32 = 3i;
const TRUE = true;
const FALSE = false;

@compute @workgroup_size(TWO, THREE, TWO - 1u)
fn main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    map_texture_kind(1);
    compose_of_splat();
    test_local_const();
    compose_vector_zero_val_binop();
    relational();
    packed_dot_product();
    test_local_const();
    abstract_access(1);
}

// Swizzle the value of nested Compose expressions.
fn swizzle_of_compose() {
    var out = vec4(vec2(1, 2), vec2(3, 4)).wzyx; // should assign vec4(4, 3, 2, 1);
}

// Index the value of nested Compose expressions.
fn index_of_compose() {
    var out = vec4(vec2(1, 2), vec2(3, 4))[1]; // should assign 2
}

// Index the value of Compose expressions nested three deep
fn compose_three_deep() {
    var out = vec4(vec3(vec2(6, 7), 8), 9)[0]; // should assign 6
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

    var out = vec4(w, x, y, z);
}

// Constant evaluation should be able to see through constants to
// their values.
const FOUR: i32 = 4;

const FOUR_ALIAS: i32 = FOUR;

const TEST_CONSTANT_ADDITION: i32 = FOUR + FOUR;
const TEST_CONSTANT_ALIAS_ADDITION: i32 = FOUR_ALIAS + FOUR_ALIAS;

fn splat_of_constant() {
    var out = -vec4(FOUR);
}

fn compose_of_constant() {
    var out = -vec4(FOUR, FOUR, FOUR, FOUR);
}

const PI: f32 = 3.141;
const phi_sun: f32 = PI * 2.0;

const DIV: vec4f = vec4(4.0 / 9.0, 0.0, 0.0, 0.0);

const TEXTURE_KIND_REGULAR: i32 = 0;
const TEXTURE_KIND_WARP: i32 = 1;
const TEXTURE_KIND_SKY: i32 = 2;

fn map_texture_kind(texture_kind: i32) -> u32 {
    switch (texture_kind) {
        case TEXTURE_KIND_REGULAR: { return 10u; }
        case TEXTURE_KIND_WARP: { return 20u; }
        case TEXTURE_KIND_SKY: { return 30u; }
        default: { return 0u; }
    }
}

fn compose_of_splat() {
    var x = vec4f(vec3f(1.0), 2.0).wzyx;
}

const add_vec = vec2(1.0f) + vec2(3.0f, 4.0f);
const compare_vec = vec2(3.0f) == vec2(3.0f, 4.0f);

// Ensure binary ops correctly flatten compositions of vector zero values
fn compose_vector_zero_val_binop() {
    var a = vec3(vec2i(), 0) + vec3(1);
    var b = vec3(vec2i(), 0) + vec3(0, 1, 2);
    var c = vec3(vec2i(), 2) + vec3(1, vec2i());
}

fn relational() {
    // Test scalar and vector forms of any() and all(), with a mixture of
    // consts, literals, zero-values, composes, and splats.
    var scalar_any_false = any(false);
    var scalar_any_true  = any(true);
    var scalar_all_false = all(false);
    var scalar_all_true  = all(true);
    var vec_any_false    = any(vec4<bool>());
    var vec_any_true     = any(vec4(bool(), true, vec2(FALSE)));
    var vec_all_false    = all(vec4(vec3(vec2<bool>(), TRUE), false));
    var vec_all_true     = all(vec4(true));
}

fn packed_dot_product() {
    // Test dot product of packed vectors on literals, constants, and
    // combinations thereof.
    var signed_four = dot4I8Packed(TWO, TWO);
    var unsigned_four = dot4U8Packed(TWO, TWO);
    var signed_twelve = dot4I8Packed(TWO + 1u, TWO + 2u);
    var unsigned_twelve = dot4U8Packed(TWO + 1u, TWO + 2u);
    var signed_seventy = dot4I8Packed(0x01020304u, 0x05060708u);
    var unsigned_seventy = dot4U8Packed(0x01020304u, 0x05060708u);

    // This is equivalent to `dot(vec4(-1, 2, -3, 4), vec4(5, 6, -7, -8))`.
    var minus_four = dot4I8Packed(0xff02fd04u, 0x0506f9f8u);
}

fn test_local_const() {
    const local_const = 2;
	var arr: array<f32, local_const>;
}

const ABSTRACT_ARRAY = array(1, 2, 3, 4, 5, 6, 7, 8, 9);
const ABSTRACT_VECTOR = vec4(1, 2, 3, 4);

fn abstract_access(i: u32) {
    // Constant indexing of abstract types is allowed, therefore we can assign
    // to f32 or u32 vars just fine.
    var a: f32 = ABSTRACT_ARRAY[0];
    var b: u32 = ABSTRACT_VECTOR.x;

    // For non constant indices the base type is concretized prior to indexing,
    // therefore we can only assign to i32 in this case.
    var c: i32 = ABSTRACT_ARRAY[i];
    var d: i32 = ABSTRACT_VECTOR[i];
}
