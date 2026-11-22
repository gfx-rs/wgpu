// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    float inner[64];
};
struct type_2 {
    float inner[8];
};

kernel void main_(
  device type_1 const& input [[buffer(0)]]
, device type_2& output [[buffer(1)]]
) {
    uint t = 0u;
    metal::float4 acc_noinit = {};
    metal::float4 acc_init = {};
    uint d = {};
    metal::uint2 loop_bound = {};
    metal::uint2 loop_bound_1 = {};
    loop_bound = metal::uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e47 = t;
            t = _e47 + 1u;
        }
        loop_init = false;
        metal::uint2 _e70 = loop_bound;
        if (metal::all(_e70 == metal::uint2(0u))) {
            break;
        }
        loop_bound = _e70 - metal::uint2((_e70.y == 0u) ? 1u : 0u, 1u);
        uint _e2 = t;
        if (_e2 < 4u) {
        } else {
            break;
        }
        {
            acc_noinit = metal::float4 {};
            acc_init = metal::float4 {};
            d = 0u;
            loop_bound_1 = metal::uint2(4294967295u);
            bool loop_init_1 = true;
            while(true) {
                if (!loop_init_1) {
                    uint _e28 = d;
                    d = _e28 + 1u;
                }
                loop_init_1 = false;
                metal::uint2 _e55 = loop_bound_1;
                if (metal::all(_e55 == metal::uint2(0u))) {
                    break;
                }
                loop_bound_1 = _e55 - metal::uint2((_e55.y == 0u) ? 1u : 0u, 1u);
                uint _e11 = d;
                if (_e11 < 16u) {
                } else {
                    break;
                }
                {
                    uint _e15 = t;
                    uint _e18 = d;
                    float _e21 = input.inner[(_e15 * 16u) + _e18];
                    metal::float4 v = metal::float4(_e21);
                    metal::float4 _e23 = acc_noinit;
                    acc_noinit = _e23 + v;
                    metal::float4 _e25 = acc_init;
                    acc_init = _e25 + v;
                }
            }
            uint _e31 = t;
            float _e36 = acc_noinit.x;
            output.inner[_e31 * 2u] = _e36;
            uint _e38 = t;
            float _e45 = acc_init.x;
            output.inner[(_e38 * 2u) + 1u] = _e45;
        }
    }
    return;
}
