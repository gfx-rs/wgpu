// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_5 {
    metal::float2x2 inner[2];
};
struct Packed {
    metal::packed_float3 member;
    float member_1;
};
struct type_6 {
    Packed inner[2];
};
constant metal::float2 VEC = metal::float2(1.0, 2.0);
constant metal::float2x2 MATRIX = metal::float2x2(VEC, VEC);
constant type_5 MATRICES = type_5 {{MATRIX, MATRIX}};
constant Packed PACKED = Packed {metal::float3(1.0, 1.0, 1.0), 2.0};
constant type_6 PACKED_ARRAY = type_6 {{PACKED, PACKED}};

void main_1(
    thread metal::float4& color
) {
    color = metal::float4(MATRICES.inner[1][0].y, PACKED_ARRAY.inner[1].member[2], PACKED_ARRAY.inner[0].member_1, 1.0);
    return;
}

struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
) {
    metal::float4 color = {};
    main_1(color);
    metal::float4 _e1 = color;
    return main_Output { _e1 };
}
