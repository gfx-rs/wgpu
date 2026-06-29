// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_4 {
    metal::atomic_int inner[2];
};
struct AtomicStruct {
    metal::atomic_uint atomic_scalar;
    type_4 atomic_arr;
};

struct test_atomic_workgroup_uniform_loadInput {
};
kernel void test_atomic_workgroup_uniform_load(
  metal::uint3 workgroup_id [[threadgroup_position_in_grid]]
, metal::uint3 local_id [[thread_position_in_threadgroup]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup metal::atomic_uint* wg_scalar
, threadgroup metal::atomic_int* wg_signed
, threadgroup AtomicStruct* wg_struct
) {
    if (__local_invocation_index == 0u) {
        metal::atomic_store_explicit(wg_scalar, 0, metal::memory_order_relaxed);
        metal::atomic_store_explicit(wg_signed, 0, metal::memory_order_relaxed);
        metal::atomic_store_explicit(&wg_struct->atomic_scalar, 0, metal::memory_order_relaxed);
        for (int __i0 = 0; __i0 < 2; __i0++) {
            metal::atomic_store_explicit(&wg_struct->atomic_arr.inner[__i0], 0, metal::memory_order_relaxed);
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    bool local = {};
    bool local_1 = {};
    bool local_2 = {};
    uint active_tile_index = workgroup_id.x + (workgroup_id.y * 32768u);
    uint _e11 = metal::atomic_fetch_or_explicit(wg_scalar, static_cast<uint>(active_tile_index >= 64u), metal::memory_order_relaxed);
    int _e14 = metal::atomic_fetch_add_explicit(wg_signed, 1, metal::memory_order_relaxed);
    metal::atomic_store_explicit(&wg_struct->atomic_scalar, 1u, metal::memory_order_relaxed);
    int _e22 = metal::atomic_fetch_add_explicit(&wg_struct->atomic_arr.inner[0], 1, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint unnamed = metal::atomic_load_explicit(wg_scalar, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int unnamed_1 = metal::atomic_load_explicit(wg_signed, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint unnamed_2 = metal::atomic_load_explicit(&wg_struct->atomic_scalar, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int unnamed_3 = metal::atomic_load_explicit(&wg_struct->atomic_arr.inner[0], metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (unnamed == 0u) {
        local = unnamed_1 > 0;
    } else {
        local = false;
    }
    bool _e41 = local;
    if (_e41) {
        local_1 = unnamed_2 > 0u;
    } else {
        local_1 = false;
    }
    bool _e47 = local_1;
    if (_e47) {
        local_2 = unnamed_3 > 0;
    } else {
        local_2 = false;
    }
    bool _e53 = local_2;
    if (_e53) {
        return;
    } else {
        return;
    }
}
