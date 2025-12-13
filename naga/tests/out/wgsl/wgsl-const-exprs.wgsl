const TWO: u32 = 2u;
const THREE: i32 = 3i;
const TRUE: bool = true;
const FALSE: bool = false;
const FOUR: i32 = 4i;
const TEXTURE_KIND_REGULAR: i32 = 0i;
const TEXTURE_KIND_WARP: i32 = 1i;
const TEXTURE_KIND_SKY: i32 = 2i;
const FOUR_ALIAS: i32 = 4i;
const TEST_CONSTANT_ADDITION: i32 = 8i;
const TEST_CONSTANT_ALIAS_ADDITION: i32 = 8i;
const PI: f32 = 3.141f;
const phi_sun: f32 = 6.282f;
const DIV: vec4<f32> = vec4<f32>(0.44444445f, 0f, 0f, 0f);
const add_vec: vec2<f32> = vec2<f32>(4f, 5f);
const compare_vec: vec2<bool> = vec2<bool>(true, false);

fn swizzle_of_compose() {
    var out: vec4<i32> = vec4<i32>(4i, 3i, 2i, 1i);

    return;
}

fn index_of_compose() {
    var out_1: i32 = 2i;

    return;
}

fn compose_three_deep() {
    var out_2: i32 = 6i;

    return;
}

fn non_constant_initializers() {
    var w: i32 = 30i;
    var x: i32;
    var y: i32;
    var z: i32 = 70i;
    var out_3: vec4<i32>;

    let _e2 = w;
    x = _e2;
    let _e4 = x;
    y = _e4;
    let _e8 = w;
    let _e9 = x;
    let _e10 = y;
    let _e11 = z;
    out_3 = vec4<i32>(_e8, _e9, _e10, _e11);
    return;
}

fn splat_of_constant() {
    var out_4: vec4<i32> = vec4<i32>(-4i, -4i, -4i, -4i);

    return;
}

fn compose_of_constant() {
    var out_5: vec4<i32> = vec4<i32>(-4i, -4i, -4i, -4i);

    return;
}

fn map_texture_kind(texture_kind: i32) -> u32 {
    switch texture_kind {
        case 0: {
            return 10u;
        }
        case 1: {
            return 20u;
        }
        case 2: {
            return 30u;
        }
        default: {
            return 0u;
        }
    }
}

fn compose_of_splat() {
    var x_1: vec4<f32> = vec4<f32>(2f, 1f, 1f, 1f);

    return;
}

fn test_local_const() {
    var arr: array<f32, 2>;

    return;
}

fn compose_vector_zero_val_binop() {
    var a: vec3<i32> = vec3<i32>(1i, 1i, 1i);
    var b: vec3<i32> = vec3<i32>(0i, 1i, 2i);
    var c: vec3<i32> = vec3<i32>(1i, 0i, 2i);

    return;
}

fn relational() {
    var scalar_any_false: bool = false;
    var scalar_any_true: bool = true;
    var scalar_all_false: bool = false;
    var scalar_all_true: bool = true;
    var vec_any_false: bool = false;
    var vec_any_true: bool = true;
    var vec_all_false: bool = false;
    var vec_all_true: bool = true;

    return;
}

fn packed_dot_product() {
    var signed_four: i32 = 4i;
    var unsigned_four: u32 = 4u;
    var signed_twelve: i32 = 12i;
    var unsigned_twelve: u32 = 12u;
    var signed_seventy: i32 = 70i;
    var unsigned_seventy: u32 = 70u;
    var minus_four: i32 = -4i;

    return;
}

fn abstract_access(i: u32) {
    var a_1: f32 = 1f;
    var b_1: u32 = 1u;
    var c_1: i32;
    var d: i32;

    c_1 = array<i32, 9>(1i, 2i, 3i, 4i, 5i, 6i, 7i, 8i, 9i)[i];
    d = vec4<i32>(1i, 2i, 3i, 4i)[i];
    return;
}

@compute @workgroup_size(2, 3, 1) 
fn main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    let _e1 = map_texture_kind(1i);
    compose_of_splat();
    test_local_const();
    compose_vector_zero_val_binop();
    relational();
    packed_dot_product();
    test_local_const();
    abstract_access(1u);
    return;
}
