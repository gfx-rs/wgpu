// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

struct PackedVec3_ {
    float a;
    char _pad1[12];
    metal::packed_float3 c;
    float d;
};
struct Matrix {
    int a;
    char _pad1[12];
    metal::int4 c;
    metal::float3x2 d;
    char _pad3[8];
};
struct TrailingPad {
    metal::float2x2 m;
    float a;
    char _pad2[4];
};
typedef float type_7[1];
constant PackedVec3_ PACKED_VEC3_ = PackedVec3_ {1.0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, metal::float3(2.0), 3.0};
constant Matrix MATRIX = Matrix {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, metal::int4(3), metal::float3x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0), metal::float2(5.0, 6.0)), {0, 0, 0, 0, 0, 0, 0, 0}};
constant TrailingPad TRAILING_PAD = TrailingPad {metal::float2x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0)), 5.0, {0, 0, 0, 0}};

float packed_vec3_(
    PackedVec3_ s
) {
    return (s.a + s.c[0]) + s.d;
}

float matrix_(
    Matrix s_1
) {
    return (static_cast<float>(s_1.a) + static_cast<float>(s_1.c.x)) + s_1.d[0].x;
}

float trailing_pad(
    TrailingPad s_2
) {
    return s_2.m[0].x + s_2.a;
}

kernel void main_(
  device type_7& out [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    float _e3 = packed_vec3_(PACKED_VEC3_);
    out[0] = _e3;
    float _e7 = matrix_(MATRIX);
    out[1] = _e7;
    float _e11 = trailing_pad(TRAILING_PAD);
    out[2] = _e11;
    return;
}
