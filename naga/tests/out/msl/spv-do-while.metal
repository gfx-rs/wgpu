// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void f_u0028_b1_u003b(
    thread bool& cond
) {
    metal::uint2 loop_bound = {};
    loop_bound = metal::uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e1 = cond;
            if (!(_e1)) {
                break;
            }
        }
        loop_init = false;
        metal::uint2 _e9 = loop_bound;
        if (metal::all(_e9 == metal::uint2(0u))) {
            break;
        }
        loop_bound = _e9 - metal::uint2((_e9.y == 0u) ? 1u : 0u, 1u);
        continue;
    }
    return;
}

void main_1(
) {
    bool param = {};
    param = false;
    f_u0028_b1_u003b(param);
    return;
}

fragment void main_(
) {
    main_1();
}
