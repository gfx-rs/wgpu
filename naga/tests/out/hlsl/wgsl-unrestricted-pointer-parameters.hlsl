struct S {
    uint a;
    uint b[4];
};

RWByteAddressBuffer buf : register(u0);
RWByteAddressBuffer arr : register(u1);
ByteAddressBuffer lut : register(t2);
Texture2D<float4> t : register(t3);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerState s = nagaSamplerHeap[nagaGroup0SamplerIndexArray[4]];
groupshared uint wg[64];
static uint private_arr[4] = (uint[4])0;

struct FragmentInput_frag {
    float2 uv_1 : LOC0;
};

typedef uint ret_ZeroValuearray2_uint_[2];
ret_ZeroValuearray2_uint_ ZeroValuearray2_uint_() {
    return (uint[2])0;
}

uint ZeroValueuint() {
    return (uint)0;
}

[numthreads(1, 1, 1)]
void main(uint index : SV_GroupIndex)
{
    uint local[2] = ZeroValuearray2_uint_();
    uint i = 0u;
    uint whole = 3u;
    uint v = (uint)0;
    uint v_1 = (uint)0;
    uint v_2 = (uint)0;
    uint local_1 = (uint)0;
    uint v_3 = (uint)0;
    uint local_2 = (uint)0;
    uint needle = (uint)0;
    int i_1 = (int)0;
    int local_3 = (int)0;
    bool local_4 = (bool)0;
    uint local_5 = (uint)0;
    bool local_6 = (bool)0;
    uint count = (uint)0;
    uint count_1 = (uint)0;
    bool c = (bool)0;
    uint local_7 = (uint)0;
    uint2 loop_bound = (4294967295u).xx;
    uint2 loop_bound_1 = (4294967295u).xx;
    uint zero_init_index = (uint)0;

    zero_init_index = index;
    while(true) {
        uint _e2 = zero_init_index;
        if ((_e2 >= 64u)) {
            break;
        }
        wg[min(uint(_e2), 63u)] = 0u;
        zero_init_index = (_e2 + 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    v = index;
    uint _e11 = v;
    buf.Store(0, asuint(_e11));
    v_1 = 1u;
    uint _e16 = v_1;
    arr.Store(index*4, asuint(_e16));
    v_2 = 2u;
    uint _e21 = v_2;
    uint _e22 = asuint(buf.Load(4+4));
    buf.Store(4+4, asuint((_e22 + _e21)));
    local_1 = _e22;
    uint _e25 = local_1;
    v_3 = 3u;
    uint _e30 = v_3;
    uint _e31 = asuint(buf.Load(8+4));
    buf.Store(8+4, asuint((_e31 + _e30)));
    local_2 = _e31;
    uint _e34 = local_2;
    uint x = (_e25 + _e34);
    arr.Store(0, asuint(x));
    needle = index;
    uint _e41 = needle;
    do {
        i_1 = int(0);
        local_4 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                int _e65 = i_1;
                i_1 = asint(asuint(_e65) + asuint(int(1)));
            }
            loop_init = false;
            {
                uint2 _e45 = loop_bound;
                loop_bound = (_e45 - uint2(((_e45.y == 0u) ? 1u : 0u), 1u));
                if (all((_e45 == (0u).xx))) {
                    break;
                }
            }
            int _e55 = i_1;
            if ((_e55 < int(64))) {
            } else {
                break;
            }
            {
                int _e59 = i_1;
                uint _e61 = wg[min(uint(_e59), 63u)];
                if ((_e61 == _e41)) {
                    int _e63 = i_1;
                    local_3 = _e63;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e68 = local_4;
        if (_e68) {
            break;
        }
        local_3 = int(-1);
    } while(false);
    int _e70 = local_3;
    arr.Store(4, asuint(uint(_e70)));
    do {
        local_6 = false;
        uint _e76 = asuint(arr.Load(12));
        switch(_e76) {
            case 0u: {
                local_5 = 10u;
                local_6 = true;
                break;
            }
            case 1u: {
                arr.Store(12, asuint(2u));
                break;
            }
            default: {
                local_5 = 30u;
                local_6 = true;
                break;
            }
        }
        bool _e78 = local_6;
        if (_e78) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e80 = local_5;
    arr.Store(8, asuint(_e80));
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e115 = i;
            i = (_e115 + 1u);
        }
        loop_init_1 = false;
        {
            uint2 _e84 = loop_bound_1;
            loop_bound_1 = (_e84 - uint2(((_e84.y == 0u) ? 1u : 0u), 1u));
            if (all((_e84 == (0u).xx))) {
                break;
            }
        }
        uint _e94 = i;
        if ((_e94 < 2u)) {
        } else {
            break;
        }
        {
            uint _e98 = i;
            count = ZeroValueuint();
            count_1 = ZeroValueuint();
            count = 1u;
            uint _e101 = local[min(uint(_e98), 1u)];
            uint _e102 = count;
            local[min(uint(_e98), 1u)] = (_e101 + _e102);
            uint _e104 = count;
            count = (_e104 + 1u);
            count_1 = 1u;
            uint _e108 = local[min(uint(_e98), 1u)];
            uint _e109 = count_1;
            local[min(uint(_e98), 1u)] = (_e108 + _e109);
            uint _e111 = count_1;
            count_1 = (_e111 + 1u);
        }
    }
    private_arr[1] = 7u;
    c = (index == 0u);
    bool _e124 = c;
    do {
        if (_e124) {
            break;
        }
        arr.Store(16, asuint(5u));
    } while(false);
    uint _e128 = local[0];
    uint _e131 = private_arr[1];
    uint _e134 = whole;
    local_7 = _e134;
    uint _e136 = local_7;
    arr.Store(20, asuint(((_e128 + _e131) + _e136)));
    return;
}

uint NagaBufferLength(ByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

float4 frag(FragmentInput_frag fragmentinput_frag) : SV_Target0
{
    float2 uv = fragmentinput_frag.uv_1;
    uint i_2 = (uint)0;
    float local_8 = (float)0;
    bool local_9 = (bool)0;
    uint2 loop_bound_2 = (4294967295u).xx;

    do {
        i_2 = 0u;
        local_9 = false;
        bool loop_init_2 = true;
        while(true) {
            if (!loop_init_2) {
                uint _e26 = i_2;
                i_2 = (_e26 + 1u);
            }
            loop_init_2 = false;
            {
                uint2 _e3 = loop_bound_2;
                loop_bound_2 = (_e3 - uint2(((_e3.y == 0u) ? 1u : 0u), 1u));
                if (all((_e3 == (0u).xx))) {
                    break;
                }
            }
            uint _e14 = i_2;
            if ((_e14 < ((NagaBufferLength(lut) - 0) / 4))) {
            } else {
                break;
            }
            {
                uint _e17 = i_2;
                float _e19 = asfloat(lut.Load(_e17*4));
                if ((_e19 > 0.0)) {
                    uint _e22 = i_2;
                    float _e24 = asfloat(lut.Load(_e22*4));
                    local_8 = _e24;
                    local_9 = true;
                    break;
                }
            }
        }
        bool _e29 = local_9;
        if (_e29) {
            break;
        }
        local_8 = 0.0;
    } while(false);
    float v_4 = local_8;
    float4 _e35 = t.Sample(s, uv);
    return (_e35 * v_4);
}
