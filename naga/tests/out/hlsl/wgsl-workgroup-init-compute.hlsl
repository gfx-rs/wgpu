struct Inner {
    float4 values[5][3];
    uint atoms[19];
    int _pad2_0;
    row_major float3x2 matrix_;
    row_major float2x2 matrices[3];
    int _end_pad_0;
    int _end_pad_1;
};

struct Tagged {
    row_major float2x2 matrix_ : LOC0;
};

static const uint width = 8u;
static const uint height = 2u;
static const uint count = 37u;

groupshared Inner data[3];
groupshared uint dynamic[37];
groupshared float floats[37];
groupshared Tagged tagged;
groupshared uint scalar;
groupshared uint3 vector_;
groupshared float2x3 matrix_;
RWByteAddressBuffer output : register(u0);

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

float4 ZeroValuefloat4() {
    return (float4)0;
}

float3x2 ZeroValuefloat3x2() {
    return (float3x2)0;
}

float2x2 ZeroValuefloat2x2() {
    return (float2x2)0;
}

Tagged ZeroValueTagged() {
    return (Tagged)0;
}

uint3 ZeroValueuint3() {
    return (uint3)0;
}

float2x3 ZeroValuefloat2x3() {
    return (float2x3)0;
}

[numthreads(8, 2, 1)]
void main(uint zero_flat_index : SV_GroupIndex, uint3 group : SV_GroupID)
{
    bool local = (bool)0;
    bool ok = (bool)0;
    bool local_1 = (bool)0;
    bool local_2 = (bool)0;
    uint i = 0u;
    uint j = (uint)0;
    uint k = (uint)0;
    uint j_1 = (uint)0;
    uint j_2 = (uint)0;
    uint i_1 = 0u;
    uint i_2 = 0u;
    uint j_3 = (uint)0;
    uint k_1 = (uint)0;
    uint j_4 = (uint)0;
    uint i_3 = 0u;
    bool local_3 = (bool)0;
    uint2 loop_bound = (4294967295u).xx;
    uint2 loop_bound_1 = (4294967295u).xx;
    uint2 loop_bound_2 = (4294967295u).xx;
    uint2 loop_bound_3 = (4294967295u).xx;
    uint2 loop_bound_4 = (4294967295u).xx;
    uint2 loop_bound_5 = (4294967295u).xx;
    uint2 loop_bound_6 = (4294967295u).xx;
    uint2 loop_bound_7 = (4294967295u).xx;
    uint2 loop_bound_8 = (4294967295u).xx;
    uint2 loop_bound_9 = (4294967295u).xx;
    uint2 loop_bound_10 = (4294967295u).xx;
    uint zero_init_index = (uint)0;

    zero_init_index = zero_flat_index;
    while(true) {
        uint _e503 = zero_init_index;
        if ((_e503 >= 45u)) {
            break;
        }
        data[min(uint(naga_div(_e503, 15u)), 2u)].values[min(uint(naga_mod(naga_div(_e503, 3u), 5u)), 4u)][min(uint(naga_mod(_e503, 3u)), 2u)] = ZeroValuefloat4();
        zero_init_index = (_e503 + 16u);
    }
    zero_init_index = zero_flat_index;
    while(true) {
        uint _e523 = zero_init_index;
        if ((_e523 >= 57u)) {
            break;
        }
        data[min(uint(naga_div(_e523, 19u)), 2u)].atoms[min(uint(naga_mod(_e523, 19u)), 18u)] = 0u;
        zero_init_index = (_e523 + 16u);
    }
    if ((zero_flat_index < 3u)) {
        data[min(uint(zero_flat_index), 2u)].matrix_ = ZeroValuefloat3x2();
    }
    if ((zero_flat_index < 9u)) {
        data[min(uint(naga_div(zero_flat_index, 3u)), 2u)].matrices[min(uint(naga_mod(zero_flat_index, 3u)), 2u)] = ZeroValuefloat2x2();
    }
    zero_init_index = zero_flat_index;
    while(true) {
        uint _e555 = zero_init_index;
        if ((_e555 >= 37u)) {
            break;
        }
        dynamic[min(uint(_e555), 36u)] = 0u;
        zero_init_index = (_e555 + 16u);
    }
    zero_init_index = zero_flat_index;
    while(true) {
        uint _e564 = zero_init_index;
        if ((_e564 >= 37u)) {
            break;
        }
        floats[min(uint(_e564), 36u)] = 0.0;
        zero_init_index = (_e564 + 16u);
    }
    if ((zero_flat_index == 0u)) {
        tagged = ZeroValueTagged();
        scalar = 0u;
        vector_ = ZeroValueuint3();
        matrix_ = ZeroValuefloat2x3();
    }
    GroupMemoryBarrierWithGroupSync();
    uint _e3 = scalar;
    if ((_e3 == 0u)) {
        uint3 _e9 = vector_;
        local = all((_e9 == (0u).xxx));
    } else {
        local = false;
    }
    bool _e15 = local;
    ok = _e15;
    bool _e17 = ok;
    float3 _e20 = matrix_[0];
    if (all((_e20 == (0.0).xxx))) {
        float3 _e29 = matrix_[1];
        local_1 = all((_e29 == (0.0).xxx));
    } else {
        local_1 = false;
    }
    bool _e35 = local_1;
    ok = (_e17 & _e35);
    bool _e37 = ok;
    float2 _e41 = tagged.matrix_[0];
    if (all((_e41 == (0.0).xx))) {
        float2 _e51 = tagged.matrix_[1];
        local_2 = all((_e51 == (0.0).xx));
    } else {
        local_2 = false;
    }
    bool _e57 = local_2;
    ok = (_e37 & _e57);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e164 = i;
            i = (_e164 + 1u);
        }
        loop_init = false;
        {
            uint2 _e409 = loop_bound;
            loop_bound = (_e409 - uint2(((_e409.y == 0u) ? 1u : 0u), 1u));
            if (all((_e409 == (0u).xx))) {
                break;
            }
        }
        uint _e61 = i;
        if ((_e61 < 3u)) {
        } else {
            break;
        }
        {
            j = 0u;
            loop_bound_1 = (4294967295u).xx;
            bool loop_init_1 = true;
            while(true) {
                if (!loop_init_1) {
                    uint _e93 = j;
                    j = (_e93 + 1u);
                }
                loop_init_1 = false;
                {
                    uint2 _e367 = loop_bound_1;
                    loop_bound_1 = (_e367 - uint2(((_e367.y == 0u) ? 1u : 0u), 1u));
                    if (all((_e367 == (0u).xx))) {
                        break;
                    }
                }
                uint _e66 = j;
                if ((_e66 < 5u)) {
                } else {
                    break;
                }
                {
                    k = 0u;
                    loop_bound_2 = (4294967295u).xx;
                    bool loop_init_2 = true;
                    while(true) {
                        if (!loop_init_2) {
                            uint _e90 = k;
                            k = (_e90 + 1u);
                        }
                        loop_init_2 = false;
                        {
                            uint2 _e352 = loop_bound_2;
                            loop_bound_2 = (_e352 - uint2(((_e352.y == 0u) ? 1u : 0u), 1u));
                            if (all((_e352 == (0u).xx))) {
                                break;
                            }
                        }
                        uint _e71 = k;
                        if ((_e71 < 3u)) {
                        } else {
                            break;
                        }
                        {
                            bool _e74 = ok;
                            uint _e76 = i;
                            uint _e79 = j;
                            uint _e81 = k;
                            float4 _e83 = data[min(uint(_e76), 2u)].values[min(uint(_e79), 4u)][min(uint(_e81), 2u)];
                            ok = (_e74 & all((_e83 == (0.0).xxxx)));
                        }
                    }
                }
            }
            j_1 = 0u;
            loop_bound_3 = (4294967295u).xx;
            bool loop_init_3 = true;
            while(true) {
                if (!loop_init_3) {
                    uint _e112 = j_1;
                    j_1 = (_e112 + 1u);
                }
                loop_init_3 = false;
                {
                    uint2 _e382 = loop_bound_3;
                    loop_bound_3 = (_e382 - uint2(((_e382.y == 0u) ? 1u : 0u), 1u));
                    if (all((_e382 == (0u).xx))) {
                        break;
                    }
                }
                uint _e97 = j_1;
                if ((_e97 < 19u)) {
                } else {
                    break;
                }
                {
                    bool _e100 = ok;
                    uint _e102 = i;
                    uint _e105 = j_1;
                    uint _e107; InterlockedOr(data[min(uint(_e102), 2u)].atoms[min(uint(_e105), 18u)], 0, _e107);
                    ok = (_e100 & (_e107 == 0u));
                }
            }
            j_2 = 0u;
            loop_bound_4 = (4294967295u).xx;
            bool loop_init_4 = true;
            while(true) {
                if (!loop_init_4) {
                    uint _e161 = j_2;
                    j_2 = (_e161 + 1u);
                }
                loop_init_4 = false;
                {
                    uint2 _e397 = loop_bound_4;
                    loop_bound_4 = (_e397 - uint2(((_e397.y == 0u) ? 1u : 0u), 1u));
                    if (all((_e397 == (0u).xx))) {
                        break;
                    }
                }
                uint _e116 = j_2;
                if ((_e116 < 3u)) {
                } else {
                    break;
                }
                {
                    bool _e119 = ok;
                    uint _e121 = i;
                    uint _e124 = j_2;
                    float2 _e126 = data[min(uint(_e121), 2u)].matrix_[min(uint(_e124), 2u)];
                    ok = (_e119 & all((_e126 == (0.0).xx)));
                    bool _e132 = ok;
                    uint _e134 = i;
                    uint _e137 = j_2;
                    float2 _e140 = data[min(uint(_e134), 2u)].matrices[min(uint(_e137), 2u)][0];
                    ok = (_e132 & all((_e140 == (0.0).xx)));
                    bool _e146 = ok;
                    uint _e148 = i;
                    uint _e151 = j_2;
                    float2 _e154 = data[min(uint(_e148), 2u)].matrices[min(uint(_e151), 2u)][1];
                    ok = (_e146 & all((_e154 == (0.0).xx)));
                }
            }
        }
    }
    bool loop_init_5 = true;
    while(true) {
        if (!loop_init_5) {
            uint _e188 = i_1;
            i_1 = (_e188 + 1u);
        }
        loop_init_5 = false;
        {
            uint2 _e421 = loop_bound_5;
            loop_bound_5 = (_e421 - uint2(((_e421.y == 0u) ? 1u : 0u), 1u));
            if (all((_e421 == (0u).xx))) {
                break;
            }
        }
        uint _e168 = i_1;
        if ((_e168 < count)) {
        } else {
            break;
        }
        {
            bool _e171 = ok;
            uint _e173 = i_1;
            uint _e175 = dynamic[min(uint(_e173), 36u)];
            ok = (_e171 & (_e175 == 0u));
            bool _e179 = ok;
            uint _e181 = i_1;
            float _e183 = floats[min(uint(_e181), 36u)];
            ok = (_e179 & (_e183 == 0.0));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if ((zero_flat_index == 0u)) {
        scalar = 42u;
        vector_ = (42u).xxx;
        matrix_ = float2x3((42.0).xxx, (42.0).xxx);
        bool loop_init_6 = true;
        while(true) {
            if (!loop_init_6) {
                uint _e250 = i_2;
                i_2 = (_e250 + 1u);
            }
            loop_init_6 = false;
            {
                uint2 _e478 = loop_bound_6;
                loop_bound_6 = (_e478 - uint2(((_e478.y == 0u) ? 1u : 0u), 1u));
                if (all((_e478 == (0u).xx))) {
                    break;
                }
            }
            uint _e205 = i_2;
            if ((_e205 < 3u)) {
            } else {
                break;
            }
            {
                j_3 = 0u;
                loop_bound_7 = (4294967295u).xx;
                bool loop_init_7 = true;
                while(true) {
                    if (!loop_init_7) {
                        uint _e232 = j_3;
                        j_3 = (_e232 + 1u);
                    }
                    loop_init_7 = false;
                    {
                        uint2 _e451 = loop_bound_7;
                        loop_bound_7 = (_e451 - uint2(((_e451.y == 0u) ? 1u : 0u), 1u));
                        if (all((_e451 == (0u).xx))) {
                            break;
                        }
                    }
                    uint _e210 = j_3;
                    if ((_e210 < 5u)) {
                    } else {
                        break;
                    }
                    {
                        k_1 = 0u;
                        loop_bound_8 = (4294967295u).xx;
                        bool loop_init_8 = true;
                        while(true) {
                            if (!loop_init_8) {
                                uint _e229 = k_1;
                                k_1 = (_e229 + 1u);
                            }
                            loop_init_8 = false;
                            {
                                uint2 _e436 = loop_bound_8;
                                loop_bound_8 = (_e436 - uint2(((_e436.y == 0u) ? 1u : 0u), 1u));
                                if (all((_e436 == (0u).xx))) {
                                    break;
                                }
                            }
                            uint _e215 = k_1;
                            if ((_e215 < 3u)) {
                            } else {
                                break;
                            }
                            {
                                uint _e219 = i_2;
                                uint _e222 = j_3;
                                uint _e224 = k_1;
                                data[min(uint(_e219), 2u)].values[min(uint(_e222), 4u)][min(uint(_e224), 2u)] = (42.0).xxxx;
                            }
                        }
                    }
                }
                j_4 = 0u;
                loop_bound_9 = (4294967295u).xx;
                bool loop_init_9 = true;
                while(true) {
                    if (!loop_init_9) {
                        uint _e247 = j_4;
                        j_4 = (_e247 + 1u);
                    }
                    loop_init_9 = false;
                    {
                        uint2 _e466 = loop_bound_9;
                        loop_bound_9 = (_e466 - uint2(((_e466.y == 0u) ? 1u : 0u), 1u));
                        if (all((_e466 == (0u).xx))) {
                            break;
                        }
                    }
                    uint _e236 = j_4;
                    if ((_e236 < 19u)) {
                    } else {
                        break;
                    }
                    {
                        uint _e240 = i_2;
                        uint _e243 = j_4;
                        { uint dummy = 0; InterlockedExchange(data[min(uint(_e240), 2u)].atoms[min(uint(_e243), 18u)], 42u, dummy); }
                    }
                }
            }
        }
        bool loop_init_10 = true;
        while(true) {
            if (!loop_init_10) {
                uint _e266 = i_3;
                i_3 = (_e266 + 1u);
            }
            loop_init_10 = false;
            {
                uint2 _e490 = loop_bound_10;
                loop_bound_10 = (_e490 - uint2(((_e490.y == 0u) ? 1u : 0u), 1u));
                if (all((_e490 == (0u).xx))) {
                    break;
                }
            }
            uint _e254 = i_3;
            if ((_e254 < count)) {
            } else {
                break;
            }
            {
                uint _e258 = i_3;
                dynamic[min(uint(_e258), 36u)] = 42u;
                uint _e262 = i_3;
                floats[min(uint(_e262), 36u)] = 42.0;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    bool _e268 = ok;
    uint _e270 = scalar;
    if ((_e270 == 42u)) {
        uint3 _e276 = vector_;
        local_3 = all((_e276 == (42u).xxx));
    } else {
        local_3 = false;
    }
    bool _e282 = local_3;
    ok = (_e268 & _e282);
    bool _e284 = ok;
    float3 _e287 = matrix_[0];
    ok = (_e284 & all((_e287 == (42.0).xxx)));
    bool _e293 = ok;
    float4 _e299 = data[2].values[4][2];
    ok = (_e293 & all((_e299 == (42.0).xxxx)));
    bool _e305 = ok;
    uint _e310; InterlockedOr(data[2].atoms[18], 0, _e310);
    ok = (_e305 & (_e310 == 42u));
    bool _e314 = ok;
    uint _e318 = dynamic[36u];
    ok = (_e314 & (_e318 == 42u));
    bool _e322 = ok;
    float _e326 = floats[36u];
    ok = (_e322 & (_e326 == 42.0));
    uint index = (((group.x * width) * height) + zero_flat_index);
    uint _e338 = asuint(output.Load(index*4));
    bool _e341 = ok;
    output.Store(index*4, asuint((_e338 | (_e341 ? 2u : 1u))));
    return;
}

[numthreads(1, 1, 1)]
void single(uint local_invocation_index : SV_GroupIndex)
{
    if ((local_invocation_index == 0u)) {
        scalar = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint _e3 = scalar;
    output.Store(0, asuint(_e3));
    return;
}
