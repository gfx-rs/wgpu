// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Input {
    metal::packed_uint3 local_invocation_id;
    uint local_invocation_index;
};

struct compute1_Input {
};
kernel void compute1_(
  metal::uint3 local_invocation_id [[thread_position_in_threadgroup]]
, uint local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup uint* wg_var
) {
    if (local_invocation_index == 0u) {
        *wg_var = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const Input input = { local_invocation_id, local_invocation_index };
    *wg_var = input.local_invocation_index * 2u;
    uint _e6 = *wg_var;
    *wg_var = _e6 + input.local_invocation_id[0];
    return;
}
