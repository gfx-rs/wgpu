struct _atomic_compare_exchange_resultSint4_ {
    int old_value;
    bool exchanged;
};

struct _atomic_compare_exchange_resultUint4_ {
    uint old_value;
    bool exchanged;
};

static const uint SIZE = 128u;

RWByteAddressBuffer arr_i32_ : register(u0);
RWByteAddressBuffer arr_u32_ : register(u1);

[numthreads(1, 1, 1)]
void test_atomic_compare_exchange_i32_()
{
    uint i = 0u;
    int old = (int)0;
    bool exchanged = (bool)0;

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e27 = i;
            i = (_e27 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            int _e8 = asint(arr_i32_.Load(_e6*4));
            old = _e8;
            exchanged = false;
            uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
            while(true) {
                if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
                loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                bool _e12 = exchanged;
                if (!(_e12)) {
                } else {
                    break;
                }
                {
                    int _e14 = old;
                    int new_ = asint((asfloat(_e14) + 1.0));
                    uint _e20 = i;
                    int _e22 = old;
                    _atomic_compare_exchange_resultSint4_ _e23; arr_i32_.InterlockedCompareExchange(_e20*4, _e22, new_, _e23.old_value);
                    _e23.exchanged = (_e23.old_value == _e22);
                    old = _e23.old_value;
                    exchanged = _e23.exchanged;
                }
            }
        }
    }
    return;
}

[numthreads(1, 1, 1)]
void test_atomic_compare_exchange_u32_()
{
    uint i_1 = 0u;
    uint old_1 = (uint)0;
    bool exchanged_1 = (bool)0;

    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e27 = i_1;
            i_1 = (_e27 + 1u);
        }
        loop_init_1 = false;
        uint _e2 = i_1;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i_1;
            uint _e8 = asuint(arr_u32_.Load(_e6*4));
            old_1 = _e8;
            exchanged_1 = false;
            uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
            while(true) {
                if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
                loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
                bool _e12 = exchanged_1;
                if (!(_e12)) {
                } else {
                    break;
                }
                {
                    uint _e14 = old_1;
                    uint new_1 = asuint((asfloat(_e14) + 1.0));
                    uint _e20 = i_1;
                    uint _e22 = old_1;
                    _atomic_compare_exchange_resultUint4_ _e23; arr_u32_.InterlockedCompareExchange(_e20*4, _e22, new_1, _e23.old_value);
                    _e23.exchanged = (_e23.old_value == _e22);
                    old_1 = _e23.old_value;
                    exchanged_1 = _e23.exchanged;
                }
            }
        }
    }
    return;
}
