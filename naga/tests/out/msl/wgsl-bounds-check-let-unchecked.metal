// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

struct type_1 {
    float inner[4];
};
typedef float type_4[1];
struct Globals {
    type_1 a;
    metal::float4 v;
    metal::float3x4 m;
    type_4 d;
    char _pad4[12];
};
struct type_5 {
    float inner[2];
};

float index_array_value(
) {
    type_5 a_1 = type_5 {{1.0, 2.0}};
    return a_1.inner[1];
}

float index_vector_value(
) {
    metal::float2 v = metal::float2(1.0, 2.0);
    return v[1];
}

metal::float2 index_matrix_value(
) {
    metal::float2x2 m = metal::float2x2(metal::float2(1.0, 2.0), metal::float2(3.0, 4.0));
    return m[1];
}

float index_negative_value(
) {
    metal::float2 v_1 = metal::float2(1.0, 2.0);
    return v_1[1];
}

float index_local(
) {
    type_5 a = type_5 {{1.0, 2.0}};
    a.inner[1] = 3.0;
    float _e8 = a.inner[1];
    return _e8;
}

float index_storage(
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = globals.a.inner[3];
    float _e8 = globals.v[3];
    float _e14 = globals.m[2][3];
    return (_e4 + _e8) + _e14;
}

void set_storage(
    float value,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.a.inner[3] = value;
    globals.v[3] = value;
    globals.m[2][3] = value;
    return;
}

float index_runtime_sized_negative(
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.d[0] = 1.0;
    float _e8 = globals.d[0];
    return _e8;
}

kernel void main_(
  device Globals& globals [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    float _e0 = index_array_value();
    float _e1 = index_vector_value();
    metal::float2 _e3 = index_matrix_value();
    float _e6 = index_negative_value();
    float _e8 = index_local();
    float _e10 = index_storage(globals, _buffer_sizes);
    float _e12 = index_runtime_sized_negative(globals, _buffer_sizes);
    float sum = (((((_e0 + _e1) + _e3.x) + _e6) + _e8) + _e10) + _e12;
    set_storage(sum, globals, _buffer_sizes);
    return;
}
