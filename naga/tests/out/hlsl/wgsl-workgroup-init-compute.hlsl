typedef struct { float2 _0; float2 _1; } __mat2x2_f32;
float2 __get_col_of_mat2x2_f32(__mat2x2_f32 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat2x2_f32(__mat2x2_f32 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    }
}
void __set_el_of_mat2x2_f32(__mat2x2_f32 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    }
}

struct Inner {
    float4 values[5][3];
    uint atoms[19];
    int _pad2_0;
    float2 matrix__0; float2 matrix__1; float2 matrix__2;
    __mat2x2_f32 matrices[3];
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

float3x2 GetMatmatrix_OnInner(Inner obj) {
    return float3x2(obj.matrix__0, obj.matrix__1, obj.matrix__2);
}

void SetMatmatrix_OnInner(Inner obj, float3x2 mat) {
    obj.matrix__0 = mat[0];
    obj.matrix__1 = mat[1];
    obj.matrix__2 = mat[2];
}

void SetMatVecmatrix_OnInner(Inner obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.matrix__0 = vec; break; }
    case 1: { obj.matrix__1 = vec; break; }
    case 2: { obj.matrix__2 = vec; break; }
    }
}

void SetMatScalarmatrix_OnInner(Inner obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.matrix__0[vec_idx] = scalar; break; }
    case 1: { obj.matrix__1[vec_idx] = scalar; break; }
    case 2: { obj.matrix__2[vec_idx] = scalar; break; }
    }
}

[numthreads(8, 2, 1)]
void main(uint zero_flat_index : SV_GroupIndex, uint3 group : SV_GroupID)
{
    [loop]
    for (uint zero_flat_index_1 = zero_flat_index; zero_flat_index_1 < 45u; zero_flat_index_1 += 16u) {
        uint zero_index = (zero_flat_index_1 / 15u) % 3u;
        uint zero_index_1 = (zero_flat_index_1 / 3u) % 5u;
        uint zero_index_2 = (zero_flat_index_1 / 1u) % 3u;
        data[zero_index].values[zero_index_1][zero_index_2] = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_2 = zero_flat_index; zero_flat_index_2 < 57u; zero_flat_index_2 += 16u) {
        uint zero_index = (zero_flat_index_2 / 19u) % 3u;
        uint zero_index_3 = (zero_flat_index_2 / 1u) % 19u;
        data[zero_index].atoms[zero_index_3] = (uint)0;
    }
    [loop]
    for (uint zero_flat_index_3 = zero_flat_index; zero_flat_index_3 < 3u; zero_flat_index_3 += 16u) {
        uint zero_index = (zero_flat_index_3 / 1u) % 3u;
        data[zero_index].matrix__0 = (float2)0;
    }
    [loop]
    for (uint zero_flat_index_4 = zero_flat_index; zero_flat_index_4 < 3u; zero_flat_index_4 += 16u) {
        uint zero_index = (zero_flat_index_4 / 1u) % 3u;
        data[zero_index].matrix__1 = (float2)0;
    }
    [loop]
    for (uint zero_flat_index_5 = zero_flat_index; zero_flat_index_5 < 3u; zero_flat_index_5 += 16u) {
        uint zero_index = (zero_flat_index_5 / 1u) % 3u;
        data[zero_index].matrix__2 = (float2)0;
    }
    [loop]
    for (uint zero_flat_index_6 = zero_flat_index; zero_flat_index_6 < 9u; zero_flat_index_6 += 16u) {
        uint zero_index = (zero_flat_index_6 / 3u) % 3u;
        uint zero_index_4 = (zero_flat_index_6 / 1u) % 3u;
        data[zero_index].matrices[zero_index_4] = (__mat2x2_f32)0;
    }
    [loop]
    for (uint zero_flat_index_7 = zero_flat_index; zero_flat_index_7 < 37u; zero_flat_index_7 += 16u) {
        uint zero_index_5 = (zero_flat_index_7 / 1u) % 37u;
        dynamic[zero_index_5] = (uint)0;
    }
    [loop]
    for (uint zero_flat_index_8 = zero_flat_index; zero_flat_index_8 < 37u; zero_flat_index_8 += 16u) {
        uint zero_index_6 = (zero_flat_index_8 / 1u) % 37u;
        floats[zero_index_6] = (float)0;
    }
    if (zero_flat_index == 0) {
        tagged.matrix_ = (float2x2)0;
    }
    if (zero_flat_index == 0) {
        scalar = (uint)0;
    }
    if (zero_flat_index == 0) {
        vector_ = (uint3)0;
    }
    if (zero_flat_index == 0) {
        matrix_ = (float2x3)0;
    }
    GroupMemoryBarrierWithGroupSync();
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
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e164 = i;
            i = (_e164 + 1u);
        }
        loop_init = false;
        uint _e61 = i;
        if ((_e61 < 3u)) {
        } else {
            break;
        }
        {
            j = 0u;
            uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
            bool loop_init_1 = true;
            while(true) {
                if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
                loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                if (!loop_init_1) {
                    uint _e93 = j;
                    j = (_e93 + 1u);
                }
                loop_init_1 = false;
                uint _e66 = j;
                if ((_e66 < 5u)) {
                } else {
                    break;
                }
                {
                    k = 0u;
                    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
                    bool loop_init_2 = true;
                    while(true) {
                        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
                        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
                        if (!loop_init_2) {
                            uint _e90 = k;
                            k = (_e90 + 1u);
                        }
                        loop_init_2 = false;
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
            uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
            bool loop_init_3 = true;
            while(true) {
                if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
                loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
                if (!loop_init_3) {
                    uint _e112 = j_1;
                    j_1 = (_e112 + 1u);
                }
                loop_init_3 = false;
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
            uint2 loop_bound_4 = uint2(4294967295u, 4294967295u);
            bool loop_init_4 = true;
            while(true) {
                if (all(loop_bound_4 == uint2(0u, 0u))) { break; }
                loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
                if (!loop_init_4) {
                    uint _e161 = j_2;
                    j_2 = (_e161 + 1u);
                }
                loop_init_4 = false;
                uint _e116 = j_2;
                if ((_e116 < 3u)) {
                } else {
                    break;
                }
                {
                    bool _e119 = ok;
                    uint _e121 = i;
                    uint _e124 = j_2;
                    float2 _e126 = GetMatmatrix_OnInner(data[min(uint(_e121), 2u)])[min(uint(_e124), 2u)];
                    ok = (_e119 & all((_e126 == (0.0).xx)));
                    bool _e132 = ok;
                    uint _e134 = i;
                    uint _e137 = j_2;
                    float2 _e140 = data[min(uint(_e134), 2u)].matrices[min(uint(_e137), 2u)]._0;
                    ok = (_e132 & all((_e140 == (0.0).xx)));
                    bool _e146 = ok;
                    uint _e148 = i;
                    uint _e151 = j_2;
                    float2 _e154 = data[min(uint(_e148), 2u)].matrices[min(uint(_e151), 2u)]._1;
                    ok = (_e146 & all((_e154 == (0.0).xx)));
                }
            }
        }
    }
    uint2 loop_bound_5 = uint2(4294967295u, 4294967295u);
    bool loop_init_5 = true;
    while(true) {
        if (all(loop_bound_5 == uint2(0u, 0u))) { break; }
        loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
        if (!loop_init_5) {
            uint _e188 = i_1;
            i_1 = (_e188 + 1u);
        }
        loop_init_5 = false;
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
        uint2 loop_bound_6 = uint2(4294967295u, 4294967295u);
        bool loop_init_6 = true;
        while(true) {
            if (all(loop_bound_6 == uint2(0u, 0u))) { break; }
            loop_bound_6 -= uint2(loop_bound_6.y == 0u, 1u);
            if (!loop_init_6) {
                uint _e250 = i_2;
                i_2 = (_e250 + 1u);
            }
            loop_init_6 = false;
            uint _e205 = i_2;
            if ((_e205 < 3u)) {
            } else {
                break;
            }
            {
                j_3 = 0u;
                uint2 loop_bound_7 = uint2(4294967295u, 4294967295u);
                bool loop_init_7 = true;
                while(true) {
                    if (all(loop_bound_7 == uint2(0u, 0u))) { break; }
                    loop_bound_7 -= uint2(loop_bound_7.y == 0u, 1u);
                    if (!loop_init_7) {
                        uint _e232 = j_3;
                        j_3 = (_e232 + 1u);
                    }
                    loop_init_7 = false;
                    uint _e210 = j_3;
                    if ((_e210 < 5u)) {
                    } else {
                        break;
                    }
                    {
                        k_1 = 0u;
                        uint2 loop_bound_8 = uint2(4294967295u, 4294967295u);
                        bool loop_init_8 = true;
                        while(true) {
                            if (all(loop_bound_8 == uint2(0u, 0u))) { break; }
                            loop_bound_8 -= uint2(loop_bound_8.y == 0u, 1u);
                            if (!loop_init_8) {
                                uint _e229 = k_1;
                                k_1 = (_e229 + 1u);
                            }
                            loop_init_8 = false;
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
                uint2 loop_bound_9 = uint2(4294967295u, 4294967295u);
                bool loop_init_9 = true;
                while(true) {
                    if (all(loop_bound_9 == uint2(0u, 0u))) { break; }
                    loop_bound_9 -= uint2(loop_bound_9.y == 0u, 1u);
                    if (!loop_init_9) {
                        uint _e247 = j_4;
                        j_4 = (_e247 + 1u);
                    }
                    loop_init_9 = false;
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
        uint2 loop_bound_10 = uint2(4294967295u, 4294967295u);
        bool loop_init_10 = true;
        while(true) {
            if (all(loop_bound_10 == uint2(0u, 0u))) { break; }
            loop_bound_10 -= uint2(loop_bound_10.y == 0u, 1u);
            if (!loop_init_10) {
                uint _e266 = i_3;
                i_3 = (_e266 + 1u);
            }
            loop_init_10 = false;
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
    uint _e321 = dynamic[36u];
    ok = (_e314 & (_e321 == 42u));
    bool _e325 = ok;
    float _e332 = floats[36u];
    ok = (_e325 & (_e332 == 42.0));
    uint index = (((group.x * width) * height) + zero_flat_index);
    uint _e344 = asuint(output.Load(index*4));
    bool _e347 = ok;
    output.Store(index*4, asuint((_e344 | (_e347 ? 2u : 1u))));
    return;
}

[numthreads(1, 1, 1)]
void single(uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        scalar = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint _e3 = scalar;
    output.Store(0, asuint(_e3));
    return;
}
