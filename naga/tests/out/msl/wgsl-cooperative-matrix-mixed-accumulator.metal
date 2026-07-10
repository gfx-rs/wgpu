// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
};

typedef half type_1[1];
typedef float type_3[1];
metal::simdgroup_half8x8 NagaCooperativeLoad(const device half* ptr, int stride, bool is_row_major) {
    metal::simdgroup_half8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_float8x8 NagaCooperativeLoad(const device float* ptr, int stride, bool is_row_major) {
    metal::simdgroup_float8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_float8x8 NagaCooperativeMultiplyAdd(const thread metal::simdgroup_half8x8& a, const thread metal::simdgroup_half8x8& b, const thread metal::simdgroup_float8x8& c) {
    metal::simdgroup_float8x8 d;
    simdgroup_multiply_accumulate(d,a,b,c);
    return d;
}


[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  device type_1 const& ab [[user(fake0)]]
, device type_3& accum [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::simdgroup_float8x8 c = {};
    metal::simdgroup_half8x8 a = NagaCooperativeLoad(&ab[0], 8u, true);
    metal::simdgroup_half8x8 b = NagaCooperativeLoad(&ab[0], 8u, true);
    c = NagaCooperativeLoad(&accum[0], 8u, true);
    metal::simdgroup_float8x8 _e13 = c;
    c = NagaCooperativeMultiplyAdd(a, b, _e13);
    metal::simdgroup_float8x8 _e15 = c;
    simdgroup_store(_e15, &accum[0], 8u, 0, true);
    return;
}
