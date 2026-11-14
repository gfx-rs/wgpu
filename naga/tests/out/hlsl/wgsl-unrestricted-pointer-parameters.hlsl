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

[numthreads(64, 1, 1)]
void main(uint index : SV_GroupIndex)
{
    if (index == 0) {
        wg = (uint[64])0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint local[2] = ZeroValuearray2_uint_();
    uint i = 0u;
    uint whole = 3u;
    uint local_1 = (uint)0;
    uint local_2 = (uint)0;
    int i_1 = (int)0;
    int local_3 = (int)0;
    bool local_4 = (bool)0;
    uint local_5 = (uint)0;
    bool local_6 = (bool)0;

    buf.Store(0, asuint(index));
    arr.Store(index*4, asuint(1u));
    uint _e8 = asuint(buf.Load(4+4));
    buf.Store(4+4, asuint((_e8 + 2u)));
    local_1 = _e8;
    uint _e12 = local_1;
    uint _e16 = asuint(buf.Load(8+4));
    buf.Store(8+4, asuint((_e16 + 3u)));
    local_2 = _e16;
    uint _e20 = local_2;
    uint x = (_e12 + _e20);
    arr.Store(0, asuint(x));
    do {
        i_1 = int(0);
        local_4 = false;
        uint2 loop_bound = uint2(4294967295u, 4294967295u);
        bool loop_init = true;
        while(true) {
            if (all(loop_bound == uint2(0u, 0u))) { break; }
            loop_bound -= uint2(loop_bound.y == 0u, 1u);
            if (!loop_init) {
                int _e37 = i_1;
                i_1 = asint(asuint(_e37) + asuint(int(1)));
            }
            loop_init = false;
            int _e27 = i_1;
            if ((_e27 < int(64))) {
            } else {
                break;
            }
            {
                int _e31 = i_1;
                uint _e33 = wg[min(uint(_e31), 63u)];
                if ((_e33 == index)) {
                    int _e35 = i_1;
                    local_3 = _e35;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e40 = local_4;
        if (_e40) {
            break;
        }
        local_3 = int(-1);
    } while(false);
    int _e42 = local_3;
    arr.Store(4, asuint(uint(_e42)));
    do {
        local_6 = false;
        uint _e48 = asuint(arr.Load(12));
        switch(_e48) {
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
        bool _e50 = local_6;
        if (_e50) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e52 = local_5;
    arr.Store(8, asuint(_e52));
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e61 = i;
            i = (_e61 + 1u);
        }
        loop_init_1 = false;
        uint _e54 = i;
        if ((_e54 < 2u)) {
        } else {
            break;
        }
        {
            uint _e58 = i;
            bump_twice(local[min(uint(_e58), 1u)]);
        }
    }
    set_private(private_arr[1]);
    do {
        if ((index == 0u)) {
            break;
        }
        arr.Store(16, asuint(5u));
    } while(false);
    uint _e72 = local[0];
    uint _e75 = private_arr[1];
    const uint _e77 = not_inlined(whole);
    arr.Store(20, asuint(((_e72 + _e75) + _e77)));
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
    float v = local_7;
    float4 _e23 = t.Sample(s, uv);
    return (_e23 * v);
}
