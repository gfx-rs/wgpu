// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size6;
};

struct Particle {
    metal::float4 pos;
    metal::float4 vel;
};
struct type_4 {
    metal::atomic_uint inner[100];
};
struct WithAtomics {
    metal::atomic_uint count;
    type_4 bins;
};
struct type_6 {
    metal::float4 inner[18];
};
struct type_7 {
    type_6 inner[18];
};
struct type_8 {
    WithAtomics inner[3];
};
struct Mixed {
    metal::float4x4 transform;
    uint flag;
    char _pad2[12];
    type_7 grid;
    type_8 cells;
    char _pad4[4];
};
struct type_9 {
    float inner[16];
};
struct type_10 {
    Particle inner[256];
};
struct type_12 {
    metal::uint2 inner[1];
};
typedef float type_14[1];
struct Inputs {
    metal::packed_uint3 group;
    uint index;
};
uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}


struct main_Input {
};
kernel void main_(
  uint local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup uint& scalar
, threadgroup type_9& small
, threadgroup type_10& particles
, threadgroup type_12& single
, threadgroup Mixed& mixed
, threadgroup metal::atomic_int& counter
, device type_14& output [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    uint zero_init_index = {};
    if (local_invocation_index < 16u) {
        small.inner[local_invocation_index] = float {};
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e78 = zero_init_index;
            uint _e79 = _e78 + 32u;
            zero_init_index = _e79;
            if ((zero_init_index + 32u) >= 256u) {
                break;
            }
        }
        loop_init = false;
        uint _e72 = zero_init_index;
        particles.inner[_e72] = Particle {};
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e107 = zero_init_index;
            uint _e108 = _e107 + 32u;
            zero_init_index = _e108;
            if ((zero_init_index + 32u) >= 324u) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e96 = zero_init_index;
        mixed.grid.inner[naga_div(_e96, 18u)].inner[naga_mod(_e96, 18u)] = metal::float4 {};
    }
    if (local_invocation_index < 3u) {
        metal::atomic_store_explicit(&mixed.cells.inner[local_invocation_index].count, uint {}, metal::memory_order_relaxed);
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_2 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e132 = zero_init_index;
            uint _e133 = _e132 + 32u;
            zero_init_index = _e133;
            if ((zero_init_index + 32u) >= 300u) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e120 = zero_init_index;
        metal::atomic_store_explicit(&mixed.cells.inner[naga_div(_e120, 100u)].bins.inner[naga_mod(_e120, 100u)], uint {}, metal::memory_order_relaxed);
    }
    if (local_invocation_index == 0u) {
        scalar = uint {};
        single.inner[0u] = metal::uint2 {};
        mixed.transform = metal::float4x4 {};
        mixed.flag = uint {};
        metal::atomic_store_explicit(&counter, int {}, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint _e3 = scalar;
    float _e7 = small.inner[3];
    float _e13 = particles.inner[7].vel.y;
    uint _e18 = single.inner[0].x;
    output[0] = ((static_cast<float>(_e3) + _e7) + _e13) + static_cast<float>(_e18);
    float _e27 = mixed.transform[1].x;
    uint _e30 = mixed.flag;
    float _e38 = mixed.grid.inner[2].inner[5].z;
    output[1] = (_e27 + static_cast<float>(_e30)) + _e38;
    uint _e47 = metal::atomic_load_explicit(&mixed.cells.inner[1].bins.inner[42], metal::memory_order_relaxed);
    uint _e52 = metal::atomic_load_explicit(&mixed.cells.inner[2].count, metal::memory_order_relaxed);
    output[2] = static_cast<float>(_e47 + _e52);
    int _e58 = metal::atomic_load_explicit(&counter, metal::memory_order_relaxed);
    output[3] = static_cast<float>(_e58);
    return;
}


struct with_indexInput {
};
kernel void with_index(
  uint index [[thread_index_in_threadgroup]]
, threadgroup type_9& small
, device type_14& output [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (index < 16u) {
        small.inner[index] = float {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float _e7 = small.inner[naga_mod(index, 16u)];
    output[index] = _e7;
    return;
}


struct with_structInput {
};
kernel void with_struct(
  metal::uint3 group [[threadgroup_position_in_grid]]
, uint index_1 [[thread_index_in_threadgroup]]
, threadgroup type_10& particles
, threadgroup metal::atomic_int& counter
, device type_14& output [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    const Inputs inputs = { group, index_1 };
    uint zero_init_index_1 = {};
    zero_init_index_1 = inputs.index;
    uint2 loop_bound_3 = uint2(4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e24 = zero_init_index_1;
            uint _e25 = _e24 + 16u;
            zero_init_index_1 = _e25;
            if ((zero_init_index_1 + 16u) >= 256u) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e18 = zero_init_index_1;
        particles.inner[_e18] = Particle {};
    }
    if (inputs.index == 0u) {
        metal::atomic_store_explicit(&counter, int {}, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float _e9 = particles.inner[inputs.index].pos.x;
    int _e11 = metal::atomic_load_explicit(&counter, metal::memory_order_relaxed);
    output[inputs.index] = _e9 + static_cast<float>(_e11);
    return;
}
