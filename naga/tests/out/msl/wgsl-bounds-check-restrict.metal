// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

struct type_1 {
    float inner[10];
};
typedef float type_4[1];
struct Globals {
    type_1 a;
    char _pad1[8];
    metal::float4 v;
    metal::float3x4 m;
    type_4 d;
    char _pad4[12];
};

float index_array(
    int i,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = globals.a.inner[metal::min(unsigned(i), 9u)];
    return _e4;
}

float index_dynamic_array(
    int i_1,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = globals.d[metal::min(unsigned(i_1), (_buffer_sizes.size0 - 112 - 4) / 4)];
    return _e4;
}

float index_vector(
    int i_2,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = globals.v[metal::min(unsigned(i_2), 3u)];
    return _e4;
}

float index_vector_by_value(
    metal::float4 v,
    int i_3
) {
    return v[metal::min(unsigned(i_3), 3u)];
}

metal::float4 index_matrix(
    int i_4,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    metal::float4 _e4 = globals.m[metal::min(unsigned(i_4), 2u)];
    return _e4;
}

float index_twice(
    int i_5,
    int j,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e6 = globals.m[metal::min(unsigned(i_5), 2u)][metal::min(unsigned(j), 3u)];
    return _e6;
}

int naga_f2i32(float value) {
    return static_cast<int>(metal::clamp(value, -2147483600.0, 2147483500.0));
}

float index_expensive(
    int i_6,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e11 = globals.a.inner[metal::min(unsigned(naga_f2i32(metal::sin(static_cast<float>(i_6) / 100.0) * 100.0)), 9u)];
    return _e11;
}

float index_in_bounds(
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e3 = globals.a.inner[9];
    float _e7 = globals.v.w;
    float _e13 = globals.m[2].w;
    return (_e3 + _e7) + _e13;
}

void set_array(
    int i_7,
    float v_1,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.a.inner[metal::min(unsigned(i_7), 9u)] = v_1;
    return;
}

void set_dynamic_array(
    int i_8,
    float v_2,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.d[metal::min(unsigned(i_8), (_buffer_sizes.size0 - 112 - 4) / 4)] = v_2;
    return;
}

void set_vector(
    int i_9,
    float v_3,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.v[metal::min(unsigned(i_9), 3u)] = v_3;
    return;
}

void set_matrix(
    int i_10,
    metal::float4 v_4,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.m[metal::min(unsigned(i_10), 2u)] = v_4;
    return;
}

void set_index_twice(
    int i_11,
    int j_1,
    float v_5,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.m[metal::min(unsigned(i_11), 2u)][metal::min(unsigned(j_1), 3u)] = v_5;
    return;
}

void set_expensive(
    int i_12,
    float v_6,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.a.inner[metal::min(unsigned(naga_f2i32(metal::sin(static_cast<float>(i_12) / 100.0) * 100.0)), 9u)] = v_6;
    return;
}

void set_in_bounds(
    float v_7,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.a.inner[9] = v_7;
    globals.v.w = v_7;
    globals.m[2].w = v_7;
    return;
}

float index_dynamic_array_constant_index(
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e3 = globals.d[metal::min(unsigned(1000), (_buffer_sizes.size0 - 112 - 4) / 4)];
    return _e3;
}

void set_dynamic_array_constant_index(
    float v_8,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    globals.d[metal::min(unsigned(1000), (_buffer_sizes.size0 - 112 - 4) / 4)] = v_8;
    return;
}

kernel void main_(
  device Globals& globals [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    float _e1 = index_array(1, globals, _buffer_sizes);
    float _e3 = index_dynamic_array(1, globals, _buffer_sizes);
    float _e5 = index_vector(1, globals, _buffer_sizes);
    float _e12 = index_vector_by_value(metal::float4(2.0, 3.0, 4.0, 5.0), 6);
    metal::float4 _e14 = index_matrix(1, globals, _buffer_sizes);
    float _e17 = index_twice(1, 2, globals, _buffer_sizes);
    float _e19 = index_expensive(1, globals, _buffer_sizes);
    float _e20 = index_in_bounds(globals, _buffer_sizes);
    set_array(1, 2.0, globals, _buffer_sizes);
    set_dynamic_array(1, 2.0, globals, _buffer_sizes);
    set_vector(1, 2.0, globals, _buffer_sizes);
    set_matrix(1, metal::float4(2.0, 3.0, 4.0, 5.0), globals, _buffer_sizes);
    set_index_twice(1, 2, 1.0, globals, _buffer_sizes);
    set_expensive(1, 1.0, globals, _buffer_sizes);
    set_in_bounds(1.0, globals, _buffer_sizes);
    float _e39 = index_dynamic_array_constant_index(globals, _buffer_sizes);
    set_dynamic_array_constant_index(1.0, globals, _buffer_sizes);
    return;
}
