// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};

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
    float _e4 = uint(i) < 10 ? globals.a.inner[i] : DefaultConstructible();
    return _e4;
}

float index_dynamic_array(
    int i_1,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = uint(i_1) < 1 + (_buffer_sizes.size0 - 112 - 4) / 4 ? globals.d[i_1] : DefaultConstructible();
    return _e4;
}

float index_vector(
    int i_2,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e4 = uint(i_2) < 4 ? globals.v[i_2] : DefaultConstructible();
    return _e4;
}

float index_vector_by_value(
    metal::float4 v,
    int i_3
) {
    return uint(i_3) < 4 ? v[i_3] : DefaultConstructible();
}

metal::float4 index_matrix(
    int i_4,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    metal::float4 _e4 = uint(i_4) < 3 ? globals.m[i_4] : DefaultConstructible();
    return _e4;
}

float index_twice(
    int i_5,
    int j,
    device Globals const& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    float _e6 = uint(j) < 4 && uint(i_5) < 3 ? globals.m[i_5][j] : DefaultConstructible();
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
    int _e9 = naga_f2i32(metal::sin(static_cast<float>(i_6) / 100.0) * 100.0);
    float _e11 = uint(_e9) < 10 ? globals.a.inner[_e9] : DefaultConstructible();
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
    if (uint(i_7) < 10) {
        globals.a.inner[i_7] = v_1;
    }
    return;
}

void set_dynamic_array(
    int i_8,
    float v_2,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(i_8) < 1 + (_buffer_sizes.size0 - 112 - 4) / 4) {
        globals.d[i_8] = v_2;
    }
    return;
}

void set_vector(
    int i_9,
    float v_3,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(i_9) < 4) {
        globals.v[i_9] = v_3;
    }
    return;
}

void set_matrix(
    int i_10,
    metal::float4 v_4,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(i_10) < 3) {
        globals.m[i_10] = v_4;
    }
    return;
}

void set_index_twice(
    int i_11,
    int j_1,
    float v_5,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(j_1) < 4 && uint(i_11) < 3) {
        globals.m[i_11][j_1] = v_5;
    }
    return;
}

void set_expensive(
    int i_12,
    float v_6,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    int _e10 = naga_f2i32(metal::sin(static_cast<float>(i_12) / 100.0) * 100.0);
    if (uint(_e10) < 10) {
        globals.a.inner[_e10] = v_6;
    }
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
    float _e3 = uint(1000) < 1 + (_buffer_sizes.size0 - 112 - 4) / 4 ? globals.d[1000] : DefaultConstructible();
    return _e3;
}

void set_dynamic_array_constant_index(
    float v_8,
    device Globals& globals,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(1000) < 1 + (_buffer_sizes.size0 - 112 - 4) / 4) {
        globals.d[1000] = v_8;
    }
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
