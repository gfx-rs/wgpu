// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


[[max_total_threads_per_threadgroup(1)]] kernel void main_(
) {
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    return;
}
