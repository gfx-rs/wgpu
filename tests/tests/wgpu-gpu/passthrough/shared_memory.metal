#include <metal_stdlib>
using namespace metal;

kernel void shared_memory(device uint* output [[buffer(0)]],
                          threadgroup uint* first [[threadgroup(0)]],
                          threadgroup uint* second [[threadgroup(1)]],
                          uint lane [[thread_index_in_threadgroup]]) {
    first[lane] = lane;
    second[lane] = lane * 2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[lane] = first[63 - lane] + second[63 - lane];
}
