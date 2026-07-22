// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
  uint __local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup metal::atomic_int& a
) {
    if (__local_invocation_index == 0u) {
        metal::atomic_store_explicit(&a, 0, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int _e2 = metal::atomic_fetch_sub_explicit(&a, -1, metal::memory_order_relaxed);
    return;
}
