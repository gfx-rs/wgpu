// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_2 {
    int inner[128];
};
constant uint SIZE = 128u;

struct test_workgroupUniformLoadInput {
};
kernel void test_workgroupUniformLoad(
  metal::uint3 workgroup_id [[threadgroup_position_in_grid]]
, uint local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup type_2& arr_i32_
) {
    uint zero_init_index = {};
    zero_init_index = local_invocation_index;
    while(true) {
        uint _e11 = zero_init_index;
        if (_e11 >= 128u) {
            break;
        }
        arr_i32_.inner[_e11] = 0;
        zero_init_index = _e11 + 4u;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    int _e4 = arr_i32_.inner[workgroup_id.x];
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (_e4 > 10) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        return;
    } else {
        return;
    }
}
