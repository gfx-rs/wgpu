// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _atomic_compare_exchange_result_Uint_4_ {
    uint old_value;
    bool exchanged;
    char _pad2[3];
};

template <typename A>
_atomic_compare_exchange_result_Uint_4_ naga_atomic_compare_exchange_weak_explicit(
    device A *atomic_ptr,
    uint cmp,
    uint v
) {
    bool swapped = metal::atomic_compare_exchange_weak_explicit(
        atomic_ptr, &cmp, v,
        metal::memory_order_relaxed, metal::memory_order_relaxed
    );
    return _atomic_compare_exchange_result_Uint_4_{cmp, swapped};
}
template <typename A>
_atomic_compare_exchange_result_Uint_4_ naga_atomic_compare_exchange_weak_explicit(
    threadgroup A *atomic_ptr,
    uint cmp,
    uint v
) {
    bool swapped = metal::atomic_compare_exchange_weak_explicit(
        atomic_ptr, &cmp, v,
        metal::memory_order_relaxed, metal::memory_order_relaxed
    );
    return _atomic_compare_exchange_result_Uint_4_{cmp, swapped};
}
constant int o = 2;

kernel void f(
  uint __local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup metal::atomic_uint& a
) {
    if (__local_invocation_index == 0u) {
        metal::atomic_store_explicit(&a, 0, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    _atomic_compare_exchange_result_Uint_4_ _e5 = naga_atomic_compare_exchange_weak_explicit(&a, 2u, 1u);
    return;
}
