// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef uint type_1[1];
struct PrimeIndices {
    type_1 data;
};

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint collatz_iterations(
    uint n_base
) {
    uint n = {};
    uint i = 0u;
    n = n_base;
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e4 = n;
        if (_e4 > 1u) {
        } else {
            break;
        }
        {
            uint _e7 = n;
            if (naga_mod(_e7, 2u) == 0u) {
                uint _e12 = n;
                n = naga_div(_e12, 2u);
            } else {
                uint _e16 = n;
                n = (3u * _e16) + 1u;
            }
            uint _e20 = i;
            i = _e20 + 1u;
        }
    }
    uint _e23 = i;
    return _e23;
}

struct main_Input {
};
kernel void main_(
  metal::uint3 global_id [[thread_position_in_grid]]
, device PrimeIndices& v_indices [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    uint _e9 = v_indices.data[global_id.x];
    uint _e10 = collatz_iterations(_e9);
    v_indices.data[global_id.x] = _e10;
    return;
}
