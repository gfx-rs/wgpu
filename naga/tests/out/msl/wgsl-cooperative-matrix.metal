// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size2;
};

typedef float type_3[1];
metal::simdgroup_float8x8 NagaCooperativeLoad(const device float* ptr, int stride, bool is_row_major) {
    metal::simdgroup_float8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_float8x8 NagaCooperativeMultiplyAdd(const thread metal::simdgroup_float8x8& a, const thread metal::simdgroup_float8x8& b, const thread metal::simdgroup_float8x8& c) {
    metal::simdgroup_float8x8 d;
    simdgroup_multiply_accumulate(d,a,b,c);
    return d;
}


kernel void main_(
  device type_3& ext [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::simdgroup_float8x8 a = {};
    metal::simdgroup_float8x8 b = {};
    metal::simdgroup_float8x8 c = {};
    metal::simdgroup_float8x8 d = {};
    c = NagaCooperativeLoad(&ext[4], 8u, false);
    metal::simdgroup_float8x8 _e6 = a;
    metal::simdgroup_float8x8 _e8 = b;
    metal::simdgroup_float8x8 _e9 = c;
    d = NagaCooperativeMultiplyAdd(_e6, _e8, _e9);
    metal::simdgroup_float8x8 _e12 = d;
    simdgroup_store(_e12, &ext[0], 8u);
    metal::simdgroup_float8x8 _e16 = d;
    c = _e16;
    return;
}
