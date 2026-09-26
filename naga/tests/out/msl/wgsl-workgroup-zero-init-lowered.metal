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
  metal::uint3 id [[thread_position_in_threadgroup]]
, uint local_invocation_index [[thread_index_in_threadgroup]]
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
            uint _e92 = zero_init_index;
            uint _e93 = _e92 + 32u;
            zero_init_index = _e93;
            if (_e93 >= 256u) {
                break;
            }
        }
        loop_init = false;
        uint _e86 = zero_init_index;
        particles.inner[_e86] = Particle {};
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e121 = zero_init_index;
            uint _e122 = _e121 + 32u;
            zero_init_index = _e122;
            if (_e122 >= 324u) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e110 = zero_init_index;
        mixed.grid.inner[naga_div(_e110, 18u)].inner[naga_mod(_e110, 18u)] = metal::float4 {};
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
            uint _e146 = zero_init_index;
            uint _e147 = _e146 + 32u;
            zero_init_index = _e147;
            if (_e147 >= 300u) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e134 = zero_init_index;
        metal::atomic_store_explicit(&mixed.cells.inner[naga_div(_e134, 100u)].bins.inner[naga_mod(_e134, 100u)], uint {}, metal::memory_order_relaxed);
    }
    if (local_invocation_index == 0u) {
        scalar = uint {};
        single.inner[0u] = metal::uint2 {};
        mixed.transform = metal::float4x4 {};
        mixed.flag = uint {};
        metal::atomic_store_explicit(&counter, int {}, metal::memory_order_relaxed);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint base = (id.x + (id.y * 8u)) * 4u;
    uint _e11 = scalar;
    float _e15 = small.inner[3];
    float _e21 = particles.inner[7].vel.y;
    uint _e26 = single.inner[0].x;
    output[base] = ((static_cast<float>(_e11) + _e15) + _e21) + static_cast<float>(_e26);
    float _e37 = mixed.transform[1].x;
    uint _e40 = mixed.flag;
    float _e48 = mixed.grid.inner[2].inner[5].z;
    output[base + 1u] = (_e37 + static_cast<float>(_e40)) + _e48;
    uint _e59 = metal::atomic_load_explicit(&mixed.cells.inner[1].bins.inner[42], metal::memory_order_relaxed);
    uint _e64 = metal::atomic_load_explicit(&mixed.cells.inner[2].count, metal::memory_order_relaxed);
    output[base + 2u] = static_cast<float>(_e59 + _e64);
    int _e72 = metal::atomic_load_explicit(&counter, metal::memory_order_relaxed);
    output[base + 3u] = static_cast<float>(_e72);
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
            if (_e25 >= 256u) {
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
