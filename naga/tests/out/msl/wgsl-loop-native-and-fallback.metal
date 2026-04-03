// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef int type_1[1];

int simple_for(
) {
    int sum = 0;
    uint2 loop_bound = uint2(4294967295u);
    for(int i = 0; i < 10; i = as_type<int>(as_type<uint>(i) + as_type<uint>(1))) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        int _e7 = sum;
        int _e8 = i;
        sum = as_type<int>(as_type<uint>(_e7) + as_type<uint>(_e8));
    }
    int _e13 = sum;
    return _e13;
}

int simple_while(
) {
    int i_1 = 0;
    uint2 loop_bound_1 = uint2(4294967295u);
    while(i_1 < 10) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        int _e6 = i_1;
        i_1 = as_type<int>(as_type<uint>(_e6) + as_type<uint>(1));
    }
    int _e8 = i_1;
    return _e8;
}

int for_with_continue(
) {
    int sum_1 = 0;
    uint2 loop_bound_2 = uint2(4294967295u);
    for(int i_2 = 0; i_2 < 10; i_2 = as_type<int>(as_type<uint>(i_2) + as_type<uint>(1))) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        int _e7 = i_2;
        if (_e7 == 5) {
            continue;
        }
        int _e10 = sum_1;
        int _e11 = i_2;
        sum_1 = as_type<int>(as_type<uint>(_e10) + as_type<uint>(_e11));
    }
    int _e16 = sum_1;
    return _e16;
}

int for_with_break(
) {
    int sum_2 = 0;
    uint2 loop_bound_3 = uint2(4294967295u);
    for(int i_3 = 0; i_3 < 10; i_3 = as_type<int>(as_type<uint>(i_3) + as_type<uint>(1))) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        int _e7 = i_3;
        if (_e7 == 5) {
            break;
        }
        int _e10 = sum_2;
        int _e11 = i_3;
        sum_2 = as_type<int>(as_type<uint>(_e10) + as_type<uint>(_e11));
    }
    int _e16 = sum_2;
    return _e16;
}

int for_infinite(
) {
    int i_4 = 0;
    uint2 loop_bound_4 = uint2(4294967295u);
    for(; ; ) {
        if (metal::all(loop_bound_4 == uint2(0u))) { break; }
        loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
        int _e2 = i_4;
        if (_e2 >= 10) {
            break;
        }
        int _e6 = i_4;
        i_4 = as_type<int>(as_type<uint>(_e6) + as_type<uint>(1));
    }
    int _e8 = i_4;
    return _e8;
}

int for_no_update(
) {
    int i_5 = 0;
    uint2 loop_bound_5 = uint2(4294967295u);
    for(; i_5 < 10; ) {
        if (metal::all(loop_bound_5 == uint2(0u))) { break; }
        loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
        int _e6 = i_5;
        i_5 = as_type<int>(as_type<uint>(_e6) + as_type<uint>(1));
    }
    int _e8 = i_5;
    return _e8;
}

int while_with_break(
) {
    int i_6 = 0;
    uint2 loop_bound_6 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_6 == uint2(0u))) { break; }
        loop_bound_6 -= uint2(loop_bound_6.y == 0u, 1u);
        int _e3 = i_6;
        if (_e3 >= 10) {
            break;
        }
        int _e7 = i_6;
        i_6 = as_type<int>(as_type<uint>(_e7) + as_type<uint>(1));
    }
    int _e9 = i_6;
    return _e9;
}

int nested_loops(
) {
    int sum_3 = 0;
    int i_7 = 0;
    uint2 loop_bound_7 = uint2(4294967295u);
    while(i_7 < 3) {
        if (metal::all(loop_bound_7 == uint2(0u))) { break; }
        loop_bound_7 -= uint2(loop_bound_7.y == 0u, 1u);
        uint2 loop_bound_8 = uint2(4294967295u);
        for(int j = 0; j < 3; j = as_type<int>(as_type<uint>(j) + as_type<uint>(1))) {
            if (metal::all(loop_bound_8 == uint2(0u))) { break; }
            loop_bound_8 -= uint2(loop_bound_8.y == 0u, 1u);
            int _e12 = sum_3;
            int _e13 = i_7;
            int _e16 = j;
            sum_3 = as_type<int>(as_type<uint>(_e12) + as_type<uint>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(_e13) * as_type<uint>(3))) + as_type<uint>(_e16))));
        }
        int _e23 = i_7;
        i_7 = as_type<int>(as_type<uint>(_e23) + as_type<uint>(1));
    }
    int _e25 = sum_3;
    return _e25;
}

int for_var_outside(
) {
    int i_8 = 0;
    uint2 loop_bound_9 = uint2(4294967295u);
    for(; i_8 < 10; i_8 = as_type<int>(as_type<uint>(i_8) + as_type<uint>(1))) {
        if (metal::all(loop_bound_9 == uint2(0u))) { break; }
        loop_bound_9 -= uint2(loop_bound_9.y == 0u, 1u);
    }
    int _e8 = i_8;
    return _e8;
}

kernel void main_(
  device type_1& out [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    int _e2 = simple_for();
    out[0] = _e2;
    int _e5 = simple_while();
    out[1] = _e5;
    int _e8 = for_with_continue();
    out[2] = _e8;
    int _e11 = for_with_break();
    out[3] = _e11;
    int _e14 = for_infinite();
    out[4] = _e14;
    int _e17 = for_no_update();
    out[5] = _e17;
    int _e20 = while_with_break();
    out[6] = _e20;
    int _e23 = nested_loops();
    out[7] = _e23;
    int _e26 = for_var_outside();
    out[8] = _e26;
    return;
}
