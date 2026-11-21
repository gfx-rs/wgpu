// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    uint inner[512];
};
struct type_3 {
    metal::atomic_int inner[8];
};
struct type_4 {
    type_3 inner[8];
};
struct WStruct {
    type_1 arr;
    metal::atomic_int atom;
    type_4 atom_arr;
};
uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}


struct main_Input {
};
kernel void main_(
  uint local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup WStruct& w_mem
, device type_1& output [[buffer(0)]]
) {
    uint zero_init_index = {};
    zero_init_index = local_invocation_index;
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e8 = zero_init_index;
        if (_e8 >= 512u) {
            break;
        }
        w_mem.arr.inner[_e8] = 0u;
        zero_init_index = _e8 + 1u;
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_1 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        uint _e18 = zero_init_index;
        if (_e18 >= 64u) {
            break;
        }
        metal::atomic_store_explicit(&w_mem.atom_arr.inner[naga_div(_e18, 8u)].inner[naga_mod(_e18, 8u)], 0, metal::memory_order_relaxed);
        zero_init_index = _e18 + 1u;
    }
    if (local_invocation_index == 0u) {
        metal::atomic_store_explicit(&w_mem.atom, 0, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    type_1 _e3 = w_mem.arr;
    output = _e3;
    return;
}
