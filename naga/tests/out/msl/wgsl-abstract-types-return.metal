// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_4 {
    float inner[4];
};

int return_i32_ai(
) {
    return 1;
}

uint return_u32_ai(
) {
    return 1u;
}

float return_f32_ai(
) {
    return 1.0;
}

float return_f32_af(
) {
    return 1.0;
}

metal::float2 return_vec2f32_ai(
) {
    return metal::float2(1.0);
}

type_4 return_arrf32_ai(
) {
    return type_4 {1.0, 1.0, 1.0, 1.0};
}

float return_const_f32_const_ai(
) {
    return 1.0;
}

metal::float2 return_vec2f32_const_ai(
) {
    return metal::float2(1.0);
}

kernel void main_(
) {
    int _e0 = return_i32_ai();
    uint _e1 = return_u32_ai();
    float _e2 = return_f32_ai();
    float _e3 = return_f32_af();
    metal::float2 _e4 = return_vec2f32_ai();
    type_4 _e5 = return_arrf32_ai();
    float _e6 = return_const_f32_const_ai();
    metal::float2 _e7 = return_vec2f32_const_ai();
    return;
}
