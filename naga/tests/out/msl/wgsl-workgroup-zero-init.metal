// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_4 {
    float inner[4];
};
struct Particle {
    metal::packed_float3 position;
    metal::atomic_uint hits;
    type_4 history;
};
struct type_5 {
    uint inner[3];
};
struct type_7 {
    metal::float4 inner[8];
};
struct type_8 {
    type_4 inner[5];
};
struct type_9 {
    Particle inner[2];
};
uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}


struct main_Input {
};
kernel void main_(
  uint index [[thread_index_in_threadgroup]]
, threadgroup uint& total
, threadgroup type_5& small
, threadgroup type_7& exact
, threadgroup type_8& grid
, threadgroup type_9& particles
) {
    uint zero_init_index = {};
    if (index < 3u) {
        small.inner[index] = 0u;
    }
    exact.inner[index] = metal::float4 {};
    zero_init_index = index;
    while(true) {
        uint _e63 = zero_init_index;
        if (_e63 >= 20u) {
            break;
        }
        grid.inner[naga_div(_e63, 4u)].inner[naga_mod(_e63, 4u)] = 0.0;
        zero_init_index = _e63 + 8u;
    }
    if (index < 2u) {
        particles.inner[index].position = metal::float3 {};
        metal::atomic_store_explicit(&particles.inner[index].hits, 0u, metal::memory_order_relaxed);
    }
    particles.inner[naga_div(index, 4u)].history.inner[naga_mod(index, 4u)] = 0.0;
    if (index == 0u) {
        total = 0u;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint _e5 = small.inner[naga_mod(index, 3u)];
    float _e10 = exact.inner[index].x;
    float _e19 = grid.inner[naga_mod(index, 5u)].inner[naga_mod(index, 4u)];
    float _e27 = particles.inner[naga_mod(index, 2u)].position[0];
    float _e37 = particles.inner[naga_mod(index, 2u)].history.inner[naga_mod(index, 4u)];
    float sum = (((static_cast<float>(_e5) + _e10) + _e19) + _e27) + _e37;
    uint _e43 = metal::atomic_fetch_add_explicit(&particles.inner[0].hits, naga_f2u32(sum), metal::memory_order_relaxed);
    if (index == 0u) {
        uint _e50 = metal::atomic_load_explicit(&particles.inner[1].hits, metal::memory_order_relaxed);
        total = _e50;
        return;
    } else {
        return;
    }
}


struct no_indexInput {
};
kernel void no_index(
  uint local_invocation_index [[thread_index_in_threadgroup]]
, threadgroup uint& total
) {
    if (local_invocation_index == 0u) {
        total = 0u;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint phony = total;
    return;
}
