// language: metal3.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size6;
};

struct Padded {
    metal::packed_uint3 xyz;
    uint tail;
};
struct type_3 {
    uint inner[513];
};
struct type_5 {
    metal::uint3 inner[3];
};
typedef uint type_6[1];
constant uint W = 64u;
constant uint N = 513u;

uint sum_small(
    threadgroup uint& flag,
    threadgroup metal::uint2& pair,
    threadgroup type_5& vectors,
    threadgroup Padded& padded
) {
    uint sum_1 = {};
    uint j_3 = 0u;
    uint _e1 = flag;
    uint _e4 = pair.x;
    uint _e8 = pair.y;
    uint _e13 = padded.xyz[0];
    uint _e18 = padded.xyz[1];
    uint _e23 = padded.xyz[2];
    uint _e27 = padded.tail;
    sum_1 = (((((_e1 + _e4) + _e8) + _e13) + _e18) + _e23) + _e27;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e55 = j_3;
            j_3 = _e55 + 1u;
        }
        loop_init = false;
        uint _e32 = j_3;
        if (_e32 < 3u) {
        } else {
            break;
        }
        {
            uint _e35 = sum_1;
            uint _e37 = j_3;
            uint _e40 = vectors.inner[_e37].x;
            uint _e42 = j_3;
            uint _e45 = vectors.inner[_e42].y;
            uint _e48 = j_3;
            uint _e51 = vectors.inner[_e48].z;
            sum_1 = _e35 + ((_e40 + _e45) + _e51);
        }
    }
    uint _e57 = sum_1;
    return _e57;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  uint i [[thread_index_in_threadgroup]]
, metal::uint3 group [[threadgroup_position_in_grid]]
, device type_6& output [[buffer(0)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(1)]]
) {
    threadgroup struct {
        type_5 vectors;
        Padded padded;
        metal::uint2 pair;
        uint flag;
        type_3 scratch;
        metal::atomic_uint counter;
    } nagaWorkgroupMemory;
    threadgroup type_5& vectors = nagaWorkgroupMemory.vectors;
    threadgroup Padded& padded = nagaWorkgroupMemory.padded;
    threadgroup metal::uint2& pair = nagaWorkgroupMemory.pair;
    threadgroup uint& flag = nagaWorkgroupMemory.flag;
    threadgroup type_3& scratch = nagaWorkgroupMemory.scratch;
    threadgroup metal::atomic_uint& counter = nagaWorkgroupMemory.counter;
    if (i == 0u) {
        flag = {};
        pair = {};
        for (int __i0 = 0; __i0 < 513; __i0++) {
            scratch.inner[__i0] = {};
        }
        metal::atomic_store_explicit(&counter, 0, metal::memory_order_relaxed);
        vectors = {};
        padded = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint initial = {};
    uint j = 0u;
    uint j_1 = {};
    uint sum = 0u;
    uint j_2 = 0u;
    uint base = group.x * 4u;
    if (i == 0u) {
        uint _e7 = sum_small(flag, pair, vectors, padded);
        uint _e9 = metal::atomic_load_explicit(&counter, metal::memory_order_relaxed);
        initial = _e7 + _e9;
        uint2 loop_bound_1 = uint2(4294967295u);
        bool loop_init_1 = true;
        while(true) {
            if (metal::all(loop_bound_1 == uint2(0u))) { break; }
            loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
            if (!loop_init_1) {
                uint _e24 = j;
                j = _e24 + 1u;
            }
            loop_init_1 = false;
            uint _e14 = j;
            if (_e14 < N) {
            } else {
                break;
            }
            {
                uint _e17 = initial;
                uint _e19 = j;
                uint _e21 = scratch.inner[_e19];
                initial = _e17 + _e21;
            }
        }
        uint _e28 = initial;
        output[base] = _e28;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    j_1 = i;
    uint2 loop_bound_2 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e39 = j_1;
            j_1 = _e39 + W;
        }
        loop_init_2 = false;
        uint _e30 = j_1;
        if (_e30 < N) {
        } else {
            break;
        }
        {
            uint _e34 = j_1;
            uint _e36 = j_1;
            scratch.inner[_e34] = _e36 + group.x;
        }
    }
    uint _e44 = metal::atomic_fetch_add_explicit(&counter, 1u, metal::memory_order_relaxed);
    if (i == 0u) {
        flag = 1u;
        pair = metal::uint2(2u, 3u);
        vectors = type_5 {{metal::uint3(4u, 5u, 6u), metal::uint3(7u, 8u, 9u), metal::uint3(10u, 11u, 12u)}};
        padded = Padded {metal::uint3(13u, 14u, 15u), 16u};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (i == 0u) {
        uint2 loop_bound_3 = uint2(4294967295u);
        bool loop_init_3 = true;
        while(true) {
            if (metal::all(loop_bound_3 == uint2(0u))) { break; }
            loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
            if (!loop_init_3) {
                uint _e90 = j_2;
                j_2 = _e90 + 1u;
            }
            loop_init_3 = false;
            uint _e80 = j_2;
            if (_e80 < N) {
            } else {
                break;
            }
            {
                uint _e83 = sum;
                uint _e85 = j_2;
                uint _e87 = scratch.inner[_e85];
                sum = _e83 + _e87;
            }
        }
        uint _e96 = sum;
        output[base + 1u] = _e96;
        uint _e102 = metal::atomic_load_explicit(&counter, metal::memory_order_relaxed);
        output[base + 2u] = _e102;
        uint _e107 = sum_small(flag, pair, vectors, padded);
        output[base + 3u] = _e107;
        return;
    } else {
        return;
    }
}


struct padded_onlyInput {
};
[[max_total_threads_per_threadgroup(1)]] kernel void padded_only(
  metal::uint3 group_1 [[threadgroup_position_in_grid]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, device type_6& output [[buffer(0)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(1)]]
) {
    threadgroup struct {
        Padded padded;
    } nagaWorkgroupMemory_1;
    threadgroup Padded& padded = nagaWorkgroupMemory_1.padded;
    if (__local_invocation_index == 0u) {
        padded = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint base_1 = group_1.x * 4u;
    uint _e9 = padded.xyz[0];
    uint _e13 = padded.xyz[1];
    uint _e18 = padded.xyz[2];
    uint _e22 = padded.tail;
    output[base_1] = ((_e9 + _e13) + _e18) + _e22;
    padded = Padded {metal::uint3(13u, 14u, 15u), 16u};
    uint _e38 = padded.xyz[0];
    uint _e42 = padded.xyz[1];
    uint _e47 = padded.xyz[2];
    uint _e51 = padded.tail;
    output[base_1 + 1u] = ((_e38 + _e42) + _e47) + _e51;
    output[base_1 + 2u] = 0u;
    output[base_1 + 3u] = 0u;
    return;
}


struct no_workgroupInput {
};
[[max_total_threads_per_threadgroup(1)]] kernel void no_workgroup(
  metal::uint3 group_2 [[threadgroup_position_in_grid]]
, device type_6& output [[buffer(0)]]
, constant _mslBufferSizes& _buffer_sizes [[buffer(1)]]
) {
    uint base_2 = group_2.x * 4u;
    output[base_2] = group_2.x;
    output[base_2 + 1u] = 0u;
    output[base_2 + 2u] = 0u;
    output[base_2 + 3u] = 0u;
    return;
}
