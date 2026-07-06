// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void f_u0028_(
) {
    int i = {};
    int acc = {};
    i = 0;
    acc = 0;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_break = false;
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e9 = i;
            i = as_type<int>(as_type<uint>(_e9) + as_type<uint>(1));
            if (loop_break) {
                break;
            }
        }
        loop_init = false;
        int _e5 = i;
        int _e6 = acc;
        acc = as_type<int>(as_type<uint>(_e6) + as_type<uint>(_e5));
        loop_break = !((_e5 < 3));
        continue;
    }
    return;
}

void main_1(
) {
    f_u0028_();
    return;
}

fragment void main_(
) {
    main_1();
}
