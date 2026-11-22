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

void bump(inout uint p)
{
    uint count = 1u;

    uint _e3 = p;
    uint _e4 = count;
    p = (_e3 + _e4);
    uint _e6 = count;
    count = (_e6 + 1u);
    return;
}

void bump_twice(inout uint p_1)
{
    bump(p_1);
    bump(p_1);
    return;
}

void set_private(inout uint p_2)
{
    p_2 = 7u;
    return;
}

uint not_inlined(inout uint p_3)
{
    uint _e1 = p_3;
    return _e1;
}

typedef uint ret_ZeroValuearray2_uint_[2];
ret_ZeroValuearray2_uint_ ZeroValuearray2_uint_() {
    return (uint[2])0;
}

[numthreads(1, 1, 1)]
void main(uint index : SV_GroupIndex)
{
    [loop]
    for (uint zero_flat_index = index; zero_flat_index < 64u; zero_flat_index += 1u) {
        uint zero_index = zero_flat_index;
        wg[zero_index] = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
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
    bool c = (bool)0;

    v = index;
    uint _e3 = v;
    buf.Store(0, asuint(_e3));
    v_1 = 1u;
    uint _e8 = v_1;
    arr.Store(index*4, asuint(_e8));
    v_2 = 2u;
    uint _e13 = v_2;
    uint _e14 = asuint(buf.Load(4+4));
    buf.Store(4+4, asuint((_e14 + _e13)));
    local_1 = _e14;
    uint _e17 = local_1;
    v_3 = 3u;
    uint _e22 = v_3;
    uint _e23 = asuint(buf.Load(8+4));
    buf.Store(8+4, asuint((_e23 + _e22)));
    local_2 = _e23;
    uint _e26 = local_2;
    uint x = (_e17 + _e26);
    arr.Store(0, asuint(x));
    needle = index;
    uint _e33 = needle;
    do {
        i_1 = int(0);
        local_4 = false;
        uint2 loop_bound = uint2(4294967295u, 4294967295u);
        bool loop_init = true;
        while(true) {
            if (all(loop_bound == uint2(0u, 0u))) { break; }
            loop_bound -= uint2(loop_bound.y == 0u, 1u);
            if (!loop_init) {
                int _e45 = i_1;
                i_1 = asint(asuint(_e45) + asuint(int(1)));
            }
            loop_init = false;
            int _e35 = i_1;
            if ((_e35 < int(64))) {
            } else {
                break;
            }
            {
                int _e39 = i_1;
                uint _e41 = wg[min(uint(_e39), 63u)];
                if ((_e41 == _e33)) {
                    int _e43 = i_1;
                    local_3 = _e43;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e48 = local_4;
        if (_e48) {
            break;
        }
        local_3 = int(-1);
    } while(false);
    int _e50 = local_3;
    arr.Store(4, asuint(uint(_e50)));
    do {
        local_6 = false;
        uint _e56 = asuint(arr.Load(12));
        switch(_e56) {
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
        bool _e58 = local_6;
        if (_e58) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e60 = local_5;
    arr.Store(8, asuint(_e60));
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e69 = i;
            i = (_e69 + 1u);
        }
        loop_init_1 = false;
        uint _e62 = i;
        if ((_e62 < 2u)) {
        } else {
            break;
        }
        {
            uint _e66 = i;
            bump_twice(local[min(uint(_e66), 1u)]);
        }
    }
    set_private(private_arr[1]);
    c = (index == 0u);
    bool _e78 = c;
    do {
        if (_e78) {
            break;
        }
        arr.Store(16, asuint(5u));
    } while(false);
    uint _e82 = local[0];
    uint _e85 = private_arr[1];
    const uint _e87 = not_inlined(whole);
    arr.Store(20, asuint(((_e82 + _e85) + _e87)));
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
    float local_7 = (float)0;
    bool local_8 = (bool)0;

    do {
        i_2 = 0u;
        local_8 = false;
        uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
        bool loop_init_2 = true;
        while(true) {
            if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
            loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
            if (!loop_init_2) {
                uint _e14 = i_2;
                i_2 = (_e14 + 1u);
            }
            loop_init_2 = false;
            uint _e2 = i_2;
            if ((_e2 < ((NagaBufferLength(lut) - 0) / 4))) {
            } else {
                break;
            }
            {
                uint _e5 = i_2;
                float _e7 = asfloat(lut.Load(_e5*4));
                if ((_e7 > 0.0)) {
                    uint _e10 = i_2;
                    float _e12 = asfloat(lut.Load(_e10*4));
                    local_7 = _e12;
                    local_8 = true;
                    break;
                }
            }
        }
        bool _e17 = local_8;
        if (_e17) {
            break;
        }
        local_7 = 0.0;
    } while(false);
    float v_4 = local_7;
    float4 _e23 = t.Sample(s, uv);
    return (_e23 * v_4);
}
