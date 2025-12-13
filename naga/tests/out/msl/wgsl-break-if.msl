// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void breakIfEmpty(
) {
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            if (true) {
                break;
            }
        }
        loop_init = false;
    }
    return;
}

void breakIfEmptyBody(
    bool a
) {
    bool b = {};
    bool c = {};
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            b = a;
            bool _e2 = b;
            c = a != _e2;
            bool _e5 = c;
            if (a == c) {
                break;
            }
        }
        loop_init_1 = false;
    }
    return;
}

void breakIf(
    bool a_1
) {
    bool d = {};
    bool e = {};
    uint2 loop_bound_2 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            bool _e5 = e;
            if (a_1 == e) {
                break;
            }
        }
        loop_init_2 = false;
        d = a_1;
        bool _e2 = d;
        e = a_1 != _e2;
    }
    return;
}

void breakIfSeparateVariable(
) {
    uint counter = 0u;
    uint2 loop_bound_3 = uint2(4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e5 = counter;
            if (counter == 5u) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e3 = counter;
        counter = _e3 + 1u;
    }
    return;
}

kernel void main_(
) {
    breakIfEmpty();
    breakIfEmptyBody(false);
    breakIf(false);
    breakIfSeparateVariable();
    return;
}
