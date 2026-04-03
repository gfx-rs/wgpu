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
    for(int i = 0; i < 10; i = as_type<int>(as_type<uint>(i) + as_type<uint>(1))) {
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
    while(i_1 < 10) {
        int _e6 = i_1;
        i_1 = as_type<int>(as_type<uint>(_e6) + as_type<uint>(1));
    }
    int _e8 = i_1;
    return _e8;
}

int for_with_continue(
) {
    int sum_1 = 0;
    for(int i_2 = 0; i_2 < 10; i_2 = as_type<int>(as_type<uint>(i_2) + as_type<uint>(1))) {
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

int nested_for_while(
) {
    int sum_2 = 0;
    int i_3 = 0;
    while(i_3 < 3) {
        for(int j = 0; j < 3; j = as_type<int>(as_type<uint>(j) + as_type<uint>(1))) {
            int _e12 = sum_2;
            int _e13 = i_3;
            int _e16 = j;
            sum_2 = as_type<int>(as_type<uint>(_e12) + as_type<uint>(as_type<int>(as_type<uint>(as_type<int>(as_type<uint>(_e13) * as_type<uint>(3))) + as_type<uint>(_e16))));
        }
        int _e23 = i_3;
        i_3 = as_type<int>(as_type<uint>(_e23) + as_type<uint>(1));
    }
    int _e25 = sum_2;
    return _e25;
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
    int _e11 = nested_for_while();
    out[3] = _e11;
    return;
}
