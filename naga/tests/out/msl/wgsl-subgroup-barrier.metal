// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    return;
}
