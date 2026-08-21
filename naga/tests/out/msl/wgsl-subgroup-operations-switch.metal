// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}

uint lowered(
    uint sid_1
) {
    uint v = 0u;
    if (naga_mod(sid_1, 3u) == 0u) {
        uint unnamed = metal::simd_broadcast_first(sid_1);
        v = unnamed;
    } else {
        if (naga_mod(sid_1, 3u) == 1u) {
            uint unnamed_1 = metal::simd_sum(sid_1);
            v = unnamed_1;
        } else {
            uint unnamed_2 = metal::simd_broadcast_first(sid_1);
            v = unnamed_2;
        }
    }
    uint _e8 = v;
    return _e8;
}

uint kept_fallthrough(
    uint sid_2
) {
    uint v_1 = 0u;
    switch(naga_mod(sid_2, 2u)) {
        case 0u:
        default: {
            uint unnamed_3 = metal::simd_broadcast_first(sid_2);
            v_1 = unnamed_3;
            break;
        }
    }
    uint _e6 = v_1;
    return _e6;
}

uint kept_break(
    uint sid_3
) {
    uint v_2 = 0u;
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        switch(naga_mod(sid_3, 2u)) {
            case 0u: {
                uint unnamed_4 = metal::simd_broadcast_first(sid_3);
                v_2 = unnamed_4;
                break;
            }
            default: {
                v_2 = 1u;
                break;
            }
        }
        break;
    }
    uint _e7 = v_2;
    return _e7;
}

uint kept_no_subgroup_ops(
    uint sid_4
) {
    uint v_3 = 0u;
    switch(naga_mod(sid_4, 2u)) {
        case 0u: {
            v_3 = 10u;
            break;
        }
        default: {
            v_3 = 20u;
            break;
        }
    }
    uint _e7 = v_3;
    return _e7;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(32)]] kernel void main_(
  uint sid [[thread_index_in_simdgroup]]
) {
    uint _e1 = lowered(sid);
    uint _e2 = kept_fallthrough(sid);
    uint _e3 = kept_break(sid);
    uint _e4 = kept_no_subgroup_ops(sid);
    return;
}
