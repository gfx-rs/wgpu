// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef uint type_1[1];
struct Data {
    type_1 values;
};

struct main_Input {
};
kernel void main_(
  uint index [[thread_index_in_threadgroup]]
, device Data& payload [[user(fake0)]]
, device metal::atomic_uint& flag [[user(fake0)]]
, threadgroup uint& stage
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (index == 0u) {
        stage = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint spins = 0u;
    payload.values[index] = index;
    metal::threadgroup_barrier(metal::mem_flags::mem_device);
    if (index == 0u) {
        metal::atomic_store_explicit(&flag, 1u, metal::memory_order_relaxed);
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e11 = metal::atomic_load_explicit(&flag, metal::memory_order_relaxed);
        if (_e11 != 0u) {
            metal::threadgroup_barrier(metal::mem_flags::mem_device);
            uint _e18 = payload.values[0];
            stage = _e18;
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            break;
        }
        uint _e19 = spins;
        spins = _e19 + 1u;
        uint _e22 = spins;
        if (_e22 > 65536u) {
            break;
        }
    }
    return;
}
