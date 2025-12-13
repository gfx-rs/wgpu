// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_6 {
    float inner[2];
};
struct type_9 {
    int inner[9];
};
constant uint TWO = 2u;
constant int THREE = 3;
constant bool TRUE = true;
constant bool FALSE = false;
constant int FOUR = 4;
constant int TEXTURE_KIND_REGULAR = 0;
constant int TEXTURE_KIND_WARP = 1;
constant int TEXTURE_KIND_SKY = 2;
constant int FOUR_ALIAS = 4;
constant int TEST_CONSTANT_ADDITION = 8;
constant int TEST_CONSTANT_ALIAS_ADDITION = 8;
constant float PI = 3.141;
constant float phi_sun = 6.282;
constant metal::float4 DIV = metal::float4(0.44444445, 0.0, 0.0, 0.0);
constant metal::float2 add_vec = metal::float2(4.0, 5.0);
constant metal::bool2 compare_vec = metal::bool2(true, false);

void swizzle_of_compose(
) {
    metal::int4 out = metal::int4(4, 3, 2, 1);
    return;
}

void index_of_compose(
) {
    int out_1 = 2;
    return;
}

void compose_three_deep(
) {
    int out_2 = 6;
    return;
}

void non_constant_initializers(
) {
    int w = 30;
    int x = {};
    int y = {};
    int z = 70;
    metal::int4 out_3 = {};
    int _e2 = w;
    x = _e2;
    int _e4 = x;
    y = _e4;
    int _e8 = w;
    int _e9 = x;
    int _e10 = y;
    int _e11 = z;
    out_3 = metal::int4(_e8, _e9, _e10, _e11);
    return;
}

void splat_of_constant(
) {
    metal::int4 out_4 = metal::int4(-4, -4, -4, -4);
    return;
}

void compose_of_constant(
) {
    metal::int4 out_5 = metal::int4(-4, -4, -4, -4);
    return;
}

uint map_texture_kind(
    int texture_kind
) {
    switch(texture_kind) {
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

void compose_of_splat(
) {
    metal::float4 x_1 = metal::float4(2.0, 1.0, 1.0, 1.0);
    return;
}

void test_local_const(
) {
    type_6 arr = {};
    return;
}

void compose_vector_zero_val_binop(
) {
    metal::int3 a = metal::int3(1, 1, 1);
    metal::int3 b = metal::int3(0, 1, 2);
    metal::int3 c = metal::int3(1, 0, 2);
    return;
}

void relational(
) {
    bool scalar_any_false = false;
    bool scalar_any_true = true;
    bool scalar_all_false = false;
    bool scalar_all_true = true;
    bool vec_any_false = false;
    bool vec_any_true = true;
    bool vec_all_false = false;
    bool vec_all_true = true;
    return;
}

void packed_dot_product(
) {
    int signed_four = 4;
    uint unsigned_four = 4u;
    int signed_twelve = 12;
    uint unsigned_twelve = 12u;
    int signed_seventy = 70;
    uint unsigned_seventy = 70u;
    int minus_four = -4;
    return;
}

void abstract_access(
    uint i
) {
    float a_1 = 1.0;
    uint b_1 = 1u;
    int c_1 = {};
    int d = {};
    c_1 = type_9 {1, 2, 3, 4, 5, 6, 7, 8, 9}.inner[i];
    d = metal::int4(1, 2, 3, 4)[i];
    return;
}

kernel void main_(
) {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    uint _e1 = map_texture_kind(1);
    compose_of_splat();
    test_local_const();
    compose_vector_zero_val_binop();
    relational();
    packed_dot_product();
    test_local_const();
    abstract_access(1u);
    return;
}
