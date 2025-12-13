// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    float inner[2];
};
struct type_2 {
    type_1 inner[3];
};

type_1 ret_array(
) {
    return type_1 {1.0, 2.0};
}

type_2 ret_array_array(
) {
    type_1 _e0 = ret_array();
    type_1 _e1 = ret_array();
    type_1 _e2 = ret_array();
    return type_2 {_e0, _e1, _e2};
}

struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
) {
    type_2 _e0 = ret_array_array();
    return main_Output { metal::float4(_e0.inner[0].inner[0], _e0.inner[0].inner[1], 0.0, 1.0) };
}
