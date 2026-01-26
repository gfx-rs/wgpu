// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void function(
) {
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    return;
}

kernel void main_(
) {
    function();
}
