struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct _atomic_compare_exchange_resultSint8_ {
    int64_t old_value;
    bool exchanged;
    int _end_pad_0;
};

struct _atomic_compare_exchange_resultUint8_ {
    uint64_t old_value;
    bool exchanged;
    int _end_pad_0;
};

static const uint SIZE = 128u;

RWByteAddressBuffer arr_i64_ : register(u0);
RWByteAddressBuffer arr_u64_ : register(u1);

[numthreads(1, 1, 1)]
void test_atomic_compare_exchange_i64_()
{
    uint i = 0u;
    int64_t old = (int64_t)0;
    bool exchanged = (bool)0;

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e26 = i;
            i = (_e26 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            int64_t _e8 = arr_i64_.Load<int64_t>(_e6*8);
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
                    int64_t _e14 = old;
                    int64_t new_ = (_e14 + 10L);
                    uint _e19 = i;
                    int64_t _e21 = old;
                    _atomic_compare_exchange_resultSint8_ _e22; arr_i64_.InterlockedCompareExchange64(_e19*8, _e21, new_, _e22.old_value);
                    _e22.exchanged = (_e22.old_value == _e21);
                    old = _e22.old_value;
                    exchanged = _e22.exchanged;
                }
            }
        }
    }
    return;
}

[numthreads(1, 1, 1)]
void test_atomic_compare_exchange_u64_()
{
    uint i_1 = 0u;
    uint64_t old_1 = (uint64_t)0;
    bool exchanged_1 = (bool)0;

    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e26 = i_1;
            i_1 = (_e26 + 1u);
        }
        loop_init_1 = false;
        uint _e2 = i_1;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i_1;
            uint64_t _e8 = arr_u64_.Load<uint64_t>(_e6*8);
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
                    uint64_t _e14 = old_1;
                    uint64_t new_1 = (_e14 + 10uL);
                    uint _e19 = i_1;
                    uint64_t _e21 = old_1;
                    _atomic_compare_exchange_resultUint8_ _e22; arr_u64_.InterlockedCompareExchange64(_e19*8, _e21, new_1, _e22.old_value);
                    _e22.exchanged = (_e22.old_value == _e21);
                    old_1 = _e22.old_value;
                    exchanged_1 = _e22.exchanged;
                }
            }
        }
    }
    return;
}
