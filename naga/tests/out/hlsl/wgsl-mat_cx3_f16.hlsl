struct StructWithMat {
    row_major half4x3 m;
    uint16_t _end_pad_0;
};

struct StructWithArrayOfStructOfMat {
    StructWithMat a[4];
};

struct mat4x3f16_std140_ {
    half3 c_0_;
    uint16_t _pad1_0;
    half3 c_1_;
    uint16_t _pad2_0;
    half3 c_2_;
    uint16_t _pad3_0;
    half3 c_3_;
    uint16_t _end_pad_0;
};

struct StructWithMat_std140_ {
    half3 m_0_;
    uint16_t _pad1_0;
    half3 m_1_;
    uint16_t _pad2_0;
    half3 m_2_;
    uint16_t _pad3_0;
    half3 m_3_;
    uint16_t _end_pad_0;
};

struct StructWithArrayOfStructOfMat_std140_ {
    StructWithMat_std140_ a[4];
};

RWByteAddressBuffer s_m : register(u0);
cbuffer u_m : register(b1) { mat4x3f16_std140_ u_m; }
RWByteAddressBuffer s_sm : register(u0, space1);
cbuffer u_sm : register(b1, space1) { StructWithMat_std140_ u_sm; }
RWByteAddressBuffer s_am : register(u0, space2);
cbuffer u_am : register(b1, space2) { mat4x3f16_std140_ u_am[4]; }
RWByteAddressBuffer s_sasm : register(u0, space3);
cbuffer u_sasm : register(b1, space3) { StructWithArrayOfStructOfMat_std140_ u_sasm; }

StructWithMat ConstructStructWithMat(half4x3 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m = arg0;
    return ret;
}

typedef half4x3 ret_Constructarray4_half4x3_[4];
ret_Constructarray4_half4x3_ Constructarray4_half4x3_(half4x3 arg0, half4x3 arg1, half4x3 arg2, half4x3 arg3) {
    half4x3 ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef StructWithMat ret_Constructarray4_StructWithMat_[4];
ret_Constructarray4_StructWithMat_ Constructarray4_StructWithMat_(StructWithMat arg0, StructWithMat arg1, StructWithMat arg2, StructWithMat arg3) {
    StructWithMat ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

StructWithArrayOfStructOfMat ConstructStructWithArrayOfStructOfMat(StructWithMat arg0[4]) {
    StructWithArrayOfStructOfMat ret = (StructWithArrayOfStructOfMat)0;
    ret.a = arg0;
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    int idx = (int)0;
    int idx_1 = (int)0;
    int idx_2 = (int)0;
    int idx_3 = (int)0;
    mat4x3f16_std140_ v[4] = (mat4x3f16_std140_[4])0;
    half4x3 local[4] = (half4x3[4])0;
    StructWithMat_std140_ v_1[4] = (StructWithMat_std140_[4])0;
    StructWithMat local_1[4] = (StructWithMat[4])0;
    StructWithMat_std140_ v_2[4] = (StructWithMat_std140_[4])0;
    StructWithMat local_2[4] = (StructWithMat[4])0;

    idx = int(1);
    int _e2 = idx;
    idx = asint(asuint(_e2) - asuint(int(1)));
    half3 _e6 = u_m.c_0_;
    half3 _e8 = u_m.c_1_;
    half3 _e10 = u_m.c_2_;
    half3 _e12 = u_m.c_3_;
    half3 _e16 = u_m.c_0_;
    int _e17 = idx;
    half3 _e20 = u_m.c_0_;
    half3 _e22 = u_m.c_1_;
    half3 _e24 = u_m.c_2_;
    half3 _e26 = u_m.c_3_;
    half _e32 = u_m.c_0_.x;
    int _e35 = idx;
    half _e37 = u_m.c_0_[_e35];
    int _e38 = idx;
    half3 _e41 = u_m.c_0_;
    half3 _e43 = u_m.c_1_;
    half3 _e45 = u_m.c_2_;
    half3 _e47 = u_m.c_3_;
    int _e51 = idx;
    int _e52 = idx;
    half3 _e55 = u_m.c_0_;
    half3 _e57 = u_m.c_1_;
    half3 _e59 = u_m.c_2_;
    half3 _e61 = u_m.c_3_;
    {
        half4x3 _value2 = half4x3(_e6, _e8, _e10, _e12);
        s_m.Store(0, _value2[0]);
        s_m.Store(8, _value2[1]);
        s_m.Store(16, _value2[2]);
        s_m.Store(24, _value2[3]);
    }
    s_m.Store(0, _e16);
    int _e68 = idx;
    s_m.Store(_e68*8, half4x3(_e20, _e22, _e24, _e26)[min(uint(_e17), 3u)]);
    s_m.Store(0+0, _e32);
    int _e75 = idx;
    s_m.Store(_e75*2+0, _e37);
    int _e78 = idx;
    s_m.Store(0+_e78*8, half4x3(_e41, _e43, _e45, _e47)[min(uint(_e38), 3u)].x);
    int _e82 = idx;
    int _e84 = idx;
    s_m.Store(_e84*2+_e82*8, half4x3(_e55, _e57, _e59, _e61)[min(uint(_e51), 3u)][min(uint(_e52), 2u)]);
    idx_1 = int(1);
    int _e88 = idx_1;
    idx_1 = asint(asuint(_e88) - asuint(int(1)));
    StructWithMat_std140_ _e91 = u_sm;
    half3 _e100 = u_sm.m_0_;
    half3 _e102 = u_sm.m_1_;
    half3 _e104 = u_sm.m_2_;
    half3 _e106 = u_sm.m_3_;
    half3 _e110 = u_sm.m_0_;
    int _e111 = idx_1;
    half3 _e114 = u_sm.m_0_;
    half3 _e116 = u_sm.m_1_;
    half3 _e118 = u_sm.m_2_;
    half3 _e120 = u_sm.m_3_;
    half _e126 = u_sm.m_0_.x;
    int _e129 = idx_1;
    half _e131 = u_sm.m_0_[_e129];
    int _e132 = idx_1;
    half3 _e135 = u_sm.m_0_;
    half3 _e137 = u_sm.m_1_;
    half3 _e139 = u_sm.m_2_;
    half3 _e141 = u_sm.m_3_;
    int _e145 = idx_1;
    int _e146 = idx_1;
    half3 _e149 = u_sm.m_0_;
    half3 _e151 = u_sm.m_1_;
    half3 _e153 = u_sm.m_2_;
    half3 _e155 = u_sm.m_3_;
    {
        StructWithMat _value2 = ConstructStructWithMat(half4x3(_e91.m_0_, _e91.m_1_, _e91.m_2_, _e91.m_3_));
        {
            half4x3 _value3 = _value2.m;
            s_sm.Store(0+0, _value3[0]);
            s_sm.Store(0+8, _value3[1]);
            s_sm.Store(0+16, _value3[2]);
            s_sm.Store(0+24, _value3[3]);
        }
    }
    {
        half4x3 _value2 = half4x3(_e100, _e102, _e104, _e106);
        s_sm.Store(0+0, _value2[0]);
        s_sm.Store(0+8, _value2[1]);
        s_sm.Store(0+16, _value2[2]);
        s_sm.Store(0+24, _value2[3]);
    }
    s_sm.Store(0+0, _e110);
    int _e166 = idx_1;
    s_sm.Store(_e166*8+0, half4x3(_e114, _e116, _e118, _e120)[min(uint(_e111), 3u)]);
    s_sm.Store(0+0+0, _e126);
    int _e175 = idx_1;
    s_sm.Store(_e175*2+0+0, _e131);
    int _e179 = idx_1;
    s_sm.Store(0+_e179*8+0, half4x3(_e135, _e137, _e139, _e141)[min(uint(_e132), 3u)].x);
    int _e184 = idx_1;
    int _e186 = idx_1;
    s_sm.Store(_e186*2+_e184*8+0, half4x3(_e149, _e151, _e153, _e155)[min(uint(_e145), 3u)][min(uint(_e146), 2u)]);
    idx_2 = int(1);
    int _e190 = idx_2;
    idx_2 = asint(asuint(_e190) - asuint(int(1)));
    mat4x3f16_std140_ _e193[4] = u_am;
    v = _e193;
    mat4x3f16_std140_ _e195[4] = v;
    local = Constructarray4_half4x3_(half4x3(_e195[0].c_0_, _e195[0].c_1_, _e195[0].c_2_, _e195[0].c_3_), half4x3(_e195[1].c_0_, _e195[1].c_1_, _e195[1].c_2_, _e195[1].c_3_), half4x3(_e195[2].c_0_, _e195[2].c_1_, _e195[2].c_2_, _e195[2].c_3_), half4x3(_e195[3].c_0_, _e195[3].c_1_, _e195[3].c_2_, _e195[3].c_3_));
    half4x3 _e222[4] = local;
    half3 _e226 = u_am[0].c_0_;
    half3 _e228 = u_am[0].c_1_;
    half3 _e230 = u_am[0].c_2_;
    half3 _e232 = u_am[0].c_3_;
    int _e235 = idx_2;
    half3 _e238 = u_am[_e235].c_0_;
    half3 _e240 = u_am[_e235].c_1_;
    half3 _e242 = u_am[_e235].c_2_;
    half3 _e244 = u_am[_e235].c_3_;
    half3 _e249 = u_am[0].c_0_;
    int _e252 = idx_2;
    half3 _e254 = u_am[0].c_0_;
    half3 _e256 = u_am[0].c_1_;
    half3 _e258 = u_am[0].c_2_;
    half3 _e260 = u_am[0].c_3_;
    int _e264 = idx_2;
    half3 _e267 = u_am[_e264].c_0_;
    int _e269 = idx_2;
    int _e271 = idx_2;
    half3 _e273 = u_am[_e269].c_0_;
    half3 _e275 = u_am[_e269].c_1_;
    half3 _e277 = u_am[_e269].c_2_;
    half3 _e279 = u_am[_e269].c_3_;
    half _e286 = u_am[0].c_0_.x;
    int _e290 = idx_2;
    half _e292 = u_am[0].c_0_[_e290];
    int _e295 = idx_2;
    half3 _e297 = u_am[0].c_0_;
    half3 _e299 = u_am[0].c_1_;
    half3 _e301 = u_am[0].c_2_;
    half3 _e303 = u_am[0].c_3_;
    int _e309 = idx_2;
    int _e310 = idx_2;
    half3 _e312 = u_am[0].c_0_;
    half3 _e314 = u_am[0].c_1_;
    half3 _e316 = u_am[0].c_2_;
    half3 _e318 = u_am[0].c_3_;
    int _e323 = idx_2;
    half _e327 = u_am[_e323].c_0_.x;
    int _e329 = idx_2;
    int _e332 = idx_2;
    half _e334 = u_am[_e329].c_0_[_e332];
    int _e336 = idx_2;
    int _e338 = idx_2;
    half3 _e340 = u_am[_e336].c_0_;
    half3 _e342 = u_am[_e336].c_1_;
    half3 _e344 = u_am[_e336].c_2_;
    half3 _e346 = u_am[_e336].c_3_;
    int _e351 = idx_2;
    int _e353 = idx_2;
    int _e354 = idx_2;
    half3 _e356 = u_am[_e351].c_0_;
    half3 _e358 = u_am[_e351].c_1_;
    half3 _e360 = u_am[_e351].c_2_;
    half3 _e362 = u_am[_e351].c_3_;
    {
        half4x3 _value2[4] = _e222;
        {
            half4x3 _value3 = _value2[0];
            s_am.Store(0+0, _value3[0]);
            s_am.Store(0+8, _value3[1]);
            s_am.Store(0+16, _value3[2]);
            s_am.Store(0+24, _value3[3]);
        }
        {
            half4x3 _value3 = _value2[1];
            s_am.Store(32+0, _value3[0]);
            s_am.Store(32+8, _value3[1]);
            s_am.Store(32+16, _value3[2]);
            s_am.Store(32+24, _value3[3]);
        }
        {
            half4x3 _value3 = _value2[2];
            s_am.Store(64+0, _value3[0]);
            s_am.Store(64+8, _value3[1]);
            s_am.Store(64+16, _value3[2]);
            s_am.Store(64+24, _value3[3]);
        }
        {
            half4x3 _value3 = _value2[3];
            s_am.Store(96+0, _value3[0]);
            s_am.Store(96+8, _value3[1]);
            s_am.Store(96+16, _value3[2]);
            s_am.Store(96+24, _value3[3]);
        }
    }
    {
        half4x3 _value2 = half4x3(_e226, _e228, _e230, _e232);
        s_am.Store(0+0, _value2[0]);
        s_am.Store(0+8, _value2[1]);
        s_am.Store(0+16, _value2[2]);
        s_am.Store(0+24, _value2[3]);
    }
    int _e369 = idx_2;
    {
        half4x3 _value2 = half4x3(_e238, _e240, _e242, _e244);
        s_am.Store(_e369*32+0, _value2[0]);
        s_am.Store(_e369*32+8, _value2[1]);
        s_am.Store(_e369*32+16, _value2[2]);
        s_am.Store(_e369*32+24, _value2[3]);
    }
    s_am.Store(0+0, _e249);
    int _e376 = idx_2;
    s_am.Store(_e376*8+0, half4x3(_e254, _e256, _e258, _e260)[min(uint(_e252), 3u)]);
    int _e379 = idx_2;
    s_am.Store(0+_e379*32, _e267);
    int _e383 = idx_2;
    int _e385 = idx_2;
    s_am.Store(_e385*8+_e383*32, half4x3(_e273, _e275, _e277, _e279)[min(uint(_e271), 3u)]);
    s_am.Store(0+0+0, _e286);
    int _e394 = idx_2;
    s_am.Store(_e394*2+0+0, _e292);
    int _e398 = idx_2;
    s_am.Store(0+_e398*8+0, half4x3(_e297, _e299, _e301, _e303)[min(uint(_e295), 3u)].x);
    int _e403 = idx_2;
    int _e405 = idx_2;
    s_am.Store(_e405*2+_e403*8+0, half4x3(_e312, _e314, _e316, _e318)[min(uint(_e309), 3u)][min(uint(_e310), 2u)]);
    int _e408 = idx_2;
    s_am.Store(0+0+_e408*32, _e327);
    int _e413 = idx_2;
    int _e416 = idx_2;
    s_am.Store(_e416*2+0+_e413*32, _e334);
    int _e419 = idx_2;
    int _e421 = idx_2;
    s_am.Store(0+_e421*8+_e419*32, half4x3(_e340, _e342, _e344, _e346)[min(uint(_e338), 3u)].x);
    int _e425 = idx_2;
    int _e427 = idx_2;
    int _e429 = idx_2;
    s_am.Store(_e429*2+_e427*8+_e425*32, half4x3(_e356, _e358, _e360, _e362)[min(uint(_e353), 3u)][min(uint(_e354), 2u)]);
    idx_3 = int(1);
    int _e433 = idx_3;
    idx_3 = asint(asuint(_e433) - asuint(int(1)));
    StructWithArrayOfStructOfMat_std140_ _e436 = u_sasm;
    v_1 = _e436.a;
    StructWithMat_std140_ _e439[4] = v_1;
    local_1 = Constructarray4_StructWithMat_(ConstructStructWithMat(half4x3(_e439[0].m_0_, _e439[0].m_1_, _e439[0].m_2_, _e439[0].m_3_)), ConstructStructWithMat(half4x3(_e439[1].m_0_, _e439[1].m_1_, _e439[1].m_2_, _e439[1].m_3_)), ConstructStructWithMat(half4x3(_e439[2].m_0_, _e439[2].m_1_, _e439[2].m_2_, _e439[2].m_3_)), ConstructStructWithMat(half4x3(_e439[3].m_0_, _e439[3].m_1_, _e439[3].m_2_, _e439[3].m_3_)));
    StructWithMat _e470[4] = local_1;
    StructWithMat_std140_ _e474[4] = u_sasm.a;
    v_2 = _e474;
    StructWithMat_std140_ _e476[4] = v_2;
    local_2 = Constructarray4_StructWithMat_(ConstructStructWithMat(half4x3(_e476[0].m_0_, _e476[0].m_1_, _e476[0].m_2_, _e476[0].m_3_)), ConstructStructWithMat(half4x3(_e476[1].m_0_, _e476[1].m_1_, _e476[1].m_2_, _e476[1].m_3_)), ConstructStructWithMat(half4x3(_e476[2].m_0_, _e476[2].m_1_, _e476[2].m_2_, _e476[2].m_3_)), ConstructStructWithMat(half4x3(_e476[3].m_0_, _e476[3].m_1_, _e476[3].m_2_, _e476[3].m_3_)));
    StructWithMat _e507[4] = local_2;
    half3 _e512 = u_sasm.a[0].m_0_;
    half3 _e514 = u_sasm.a[0].m_1_;
    half3 _e516 = u_sasm.a[0].m_2_;
    half3 _e518 = u_sasm.a[0].m_3_;
    int _e522 = idx_3;
    half3 _e525 = u_sasm.a[_e522].m_0_;
    half3 _e527 = u_sasm.a[_e522].m_1_;
    half3 _e529 = u_sasm.a[_e522].m_2_;
    half3 _e531 = u_sasm.a[_e522].m_3_;
    half3 _e537 = u_sasm.a[0].m_0_;
    int _e541 = idx_3;
    half3 _e543 = u_sasm.a[0].m_0_;
    half3 _e545 = u_sasm.a[0].m_1_;
    half3 _e547 = u_sasm.a[0].m_2_;
    half3 _e549 = u_sasm.a[0].m_3_;
    int _e554 = idx_3;
    half3 _e557 = u_sasm.a[_e554].m_0_;
    int _e560 = idx_3;
    int _e562 = idx_3;
    half3 _e564 = u_sasm.a[_e560].m_0_;
    half3 _e566 = u_sasm.a[_e560].m_1_;
    half3 _e568 = u_sasm.a[_e560].m_2_;
    half3 _e570 = u_sasm.a[_e560].m_3_;
    half _e578 = u_sasm.a[0].m_0_.x;
    int _e583 = idx_3;
    half _e585 = u_sasm.a[0].m_0_[_e583];
    int _e589 = idx_3;
    half3 _e591 = u_sasm.a[0].m_0_;
    half3 _e593 = u_sasm.a[0].m_1_;
    half3 _e595 = u_sasm.a[0].m_2_;
    half3 _e597 = u_sasm.a[0].m_3_;
    int _e604 = idx_3;
    int _e605 = idx_3;
    half3 _e607 = u_sasm.a[0].m_0_;
    half3 _e609 = u_sasm.a[0].m_1_;
    half3 _e611 = u_sasm.a[0].m_2_;
    half3 _e613 = u_sasm.a[0].m_3_;
    int _e619 = idx_3;
    half _e623 = u_sasm.a[_e619].m_0_.x;
    int _e626 = idx_3;
    int _e629 = idx_3;
    half _e631 = u_sasm.a[_e626].m_0_[_e629];
    int _e634 = idx_3;
    int _e636 = idx_3;
    half3 _e638 = u_sasm.a[_e634].m_0_;
    half3 _e640 = u_sasm.a[_e634].m_1_;
    half3 _e642 = u_sasm.a[_e634].m_2_;
    half3 _e644 = u_sasm.a[_e634].m_3_;
    int _e650 = idx_3;
    int _e652 = idx_3;
    int _e653 = idx_3;
    half3 _e655 = u_sasm.a[_e650].m_0_;
    half3 _e657 = u_sasm.a[_e650].m_1_;
    half3 _e659 = u_sasm.a[_e650].m_2_;
    half3 _e661 = u_sasm.a[_e650].m_3_;
    {
        StructWithArrayOfStructOfMat _value2 = ConstructStructWithArrayOfStructOfMat(_e470);
        {
            StructWithMat _value3[4] = _value2.a;
            {
                StructWithMat _value4 = _value3[0];
                {
                    half4x3 _value5 = _value4.m;
                    s_sasm.Store(0+0+0+0, _value5[0]);
                    s_sasm.Store(0+0+0+8, _value5[1]);
                    s_sasm.Store(0+0+0+16, _value5[2]);
                    s_sasm.Store(0+0+0+24, _value5[3]);
                }
            }
            {
                StructWithMat _value4 = _value3[1];
                {
                    half4x3 _value5 = _value4.m;
                    s_sasm.Store(0+32+0+0, _value5[0]);
                    s_sasm.Store(0+32+0+8, _value5[1]);
                    s_sasm.Store(0+32+0+16, _value5[2]);
                    s_sasm.Store(0+32+0+24, _value5[3]);
                }
            }
            {
                StructWithMat _value4 = _value3[2];
                {
                    half4x3 _value5 = _value4.m;
                    s_sasm.Store(0+64+0+0, _value5[0]);
                    s_sasm.Store(0+64+0+8, _value5[1]);
                    s_sasm.Store(0+64+0+16, _value5[2]);
                    s_sasm.Store(0+64+0+24, _value5[3]);
                }
            }
            {
                StructWithMat _value4 = _value3[3];
                {
                    half4x3 _value5 = _value4.m;
                    s_sasm.Store(0+96+0+0, _value5[0]);
                    s_sasm.Store(0+96+0+8, _value5[1]);
                    s_sasm.Store(0+96+0+16, _value5[2]);
                    s_sasm.Store(0+96+0+24, _value5[3]);
                }
            }
        }
    }
    {
        StructWithMat _value2[4] = _e507;
        {
            StructWithMat _value3 = _value2[0];
            {
                half4x3 _value4 = _value3.m;
                s_sasm.Store(0+0+0+0, _value4[0]);
                s_sasm.Store(0+0+0+8, _value4[1]);
                s_sasm.Store(0+0+0+16, _value4[2]);
                s_sasm.Store(0+0+0+24, _value4[3]);
            }
        }
        {
            StructWithMat _value3 = _value2[1];
            {
                half4x3 _value4 = _value3.m;
                s_sasm.Store(0+32+0+0, _value4[0]);
                s_sasm.Store(0+32+0+8, _value4[1]);
                s_sasm.Store(0+32+0+16, _value4[2]);
                s_sasm.Store(0+32+0+24, _value4[3]);
            }
        }
        {
            StructWithMat _value3 = _value2[2];
            {
                half4x3 _value4 = _value3.m;
                s_sasm.Store(0+64+0+0, _value4[0]);
                s_sasm.Store(0+64+0+8, _value4[1]);
                s_sasm.Store(0+64+0+16, _value4[2]);
                s_sasm.Store(0+64+0+24, _value4[3]);
            }
        }
        {
            StructWithMat _value3 = _value2[3];
            {
                half4x3 _value4 = _value3.m;
                s_sasm.Store(0+96+0+0, _value4[0]);
                s_sasm.Store(0+96+0+8, _value4[1]);
                s_sasm.Store(0+96+0+16, _value4[2]);
                s_sasm.Store(0+96+0+24, _value4[3]);
            }
        }
    }
    {
        half4x3 _value2 = half4x3(_e512, _e514, _e516, _e518);
        s_sasm.Store(0+0+0+0, _value2[0]);
        s_sasm.Store(0+0+0+8, _value2[1]);
        s_sasm.Store(0+0+0+16, _value2[2]);
        s_sasm.Store(0+0+0+24, _value2[3]);
    }
    int _e673 = idx_3;
    {
        half4x3 _value2 = half4x3(_e525, _e527, _e529, _e531);
        s_sasm.Store(0+_e673*32+0+0, _value2[0]);
        s_sasm.Store(0+_e673*32+0+8, _value2[1]);
        s_sasm.Store(0+_e673*32+0+16, _value2[2]);
        s_sasm.Store(0+_e673*32+0+24, _value2[3]);
    }
    s_sasm.Store(0+0+0+0, _e537);
    int _e685 = idx_3;
    s_sasm.Store(_e685*8+0+0+0, half4x3(_e543, _e545, _e547, _e549)[min(uint(_e541), 3u)]);
    int _e689 = idx_3;
    s_sasm.Store(0+0+_e689*32+0, _e557);
    int _e695 = idx_3;
    int _e698 = idx_3;
    s_sasm.Store(_e698*8+0+_e695*32+0, half4x3(_e564, _e566, _e568, _e570)[min(uint(_e562), 3u)]);
    s_sasm.Store(0+0+0+0+0, _e578);
    int _e711 = idx_3;
    s_sasm.Store(_e711*2+0+0+0+0, _e585);
    int _e717 = idx_3;
    s_sasm.Store(0+_e717*8+0+0+0, half4x3(_e591, _e593, _e595, _e597)[min(uint(_e589), 3u)].x);
    int _e724 = idx_3;
    int _e726 = idx_3;
    s_sasm.Store(_e726*2+_e724*8+0+0+0, half4x3(_e607, _e609, _e611, _e613)[min(uint(_e604), 3u)][min(uint(_e605), 2u)]);
    int _e730 = idx_3;
    s_sasm.Store(0+0+0+_e730*32+0, _e623);
    int _e737 = idx_3;
    int _e741 = idx_3;
    s_sasm.Store(_e741*2+0+0+_e737*32+0, _e631);
    int _e745 = idx_3;
    int _e748 = idx_3;
    s_sasm.Store(0+_e748*8+0+_e745*32+0, half4x3(_e638, _e640, _e642, _e644)[min(uint(_e636), 3u)].x);
    int _e753 = idx_3;
    int _e756 = idx_3;
    int _e758 = idx_3;
    s_sasm.Store(_e758*2+_e756*8+0+_e753*32+0, half4x3(_e655, _e657, _e659, _e661)[min(uint(_e652), 3u)][min(uint(_e653), 2u)]);
    return;
}
