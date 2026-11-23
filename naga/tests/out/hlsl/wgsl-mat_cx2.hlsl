struct StructWithMat {
    row_major float2x2 m;
};

struct StructWithArrayOfStructOfMat {
    StructWithMat a[4];
};

struct mat2x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
};

struct StructWithMat_std140_ {
    float2 m_0_;
    float2 m_1_;
};

struct StructWithArrayOfStructOfMat_std140_ {
    StructWithMat_std140_ a[4];
};

RWByteAddressBuffer s_m : register(u0);
cbuffer u_m : register(b1) { mat2x2f32_std140_ u_m; }
RWByteAddressBuffer s_sm : register(u0, space1);
cbuffer u_sm : register(b1, space1) { StructWithMat_std140_ u_sm; }
RWByteAddressBuffer s_am : register(u0, space2);
cbuffer u_am : register(b1, space2) { mat2x2f32_std140_ u_am[4]; }
RWByteAddressBuffer s_sasm : register(u0, space3);
cbuffer u_sasm : register(b1, space3) { StructWithArrayOfStructOfMat_std140_ u_sasm; }

StructWithMat ConstructStructWithMat(float2x2 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m = arg0;
    return ret;
}

typedef float2x2 ret_Constructarray4_float2x2_[4];
ret_Constructarray4_float2x2_ Constructarray4_float2x2_(float2x2 arg0, float2x2 arg1, float2x2 arg2, float2x2 arg3) {
    float2x2 ret[4] = { arg0, arg1, arg2, arg3 };
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
    mat2x2f32_std140_ v[4] = (mat2x2f32_std140_[4])0;
    float2x2 local[4] = (float2x2[4])0;
    StructWithMat_std140_ v_1[4] = (StructWithMat_std140_[4])0;
    StructWithMat local_1[4] = (StructWithMat[4])0;
    StructWithMat_std140_ v_2[4] = (StructWithMat_std140_[4])0;
    StructWithMat local_2[4] = (StructWithMat[4])0;

    idx = int(1);
    int _e2 = idx;
    idx = asint(asuint(_e2) - asuint(int(1)));
    float2 _e6 = u_m.c_0_;
    float2 _e8 = u_m.c_1_;
    float2 _e12 = u_m.c_0_;
    int _e13 = idx;
    float2 _e16 = u_m.c_0_;
    float2 _e18 = u_m.c_1_;
    float _e24 = u_m.c_0_.x;
    int _e27 = idx;
    float _e29 = u_m.c_0_[_e27];
    int _e30 = idx;
    float2 _e33 = u_m.c_0_;
    float2 _e35 = u_m.c_1_;
    int _e39 = idx;
    int _e40 = idx;
    float2 _e43 = u_m.c_0_;
    float2 _e45 = u_m.c_1_;
    {
        float2x2 _value2 = float2x2(_e6, _e8);
        s_m.Store2(0, asuint(_value2[0]));
        s_m.Store2(8, asuint(_value2[1]));
    }
    s_m.Store2(0, asuint(_e12));
    int _e52 = idx;
    s_m.Store2(_e52*8, asuint(float2x2(_e16, _e18)[min(uint(_e13), 1u)]));
    s_m.Store(0+0, asuint(_e24));
    int _e59 = idx;
    s_m.Store(_e59*4+0, asuint(_e29));
    int _e62 = idx;
    s_m.Store(0+_e62*8, asuint(float2x2(_e33, _e35)[min(uint(_e30), 1u)].x));
    int _e66 = idx;
    int _e68 = idx;
    s_m.Store(_e68*4+_e66*8, asuint(float2x2(_e43, _e45)[min(uint(_e39), 1u)][min(uint(_e40), 1u)]));
    idx_1 = int(1);
    int _e72 = idx_1;
    idx_1 = asint(asuint(_e72) - asuint(int(1)));
    StructWithMat_std140_ _e75 = u_sm;
    float2 _e82 = u_sm.m_0_;
    float2 _e84 = u_sm.m_1_;
    float2 _e88 = u_sm.m_0_;
    int _e89 = idx_1;
    float2 _e92 = u_sm.m_0_;
    float2 _e94 = u_sm.m_1_;
    float _e100 = u_sm.m_0_.x;
    int _e103 = idx_1;
    float _e105 = u_sm.m_0_[_e103];
    int _e106 = idx_1;
    float2 _e109 = u_sm.m_0_;
    float2 _e111 = u_sm.m_1_;
    int _e115 = idx_1;
    int _e116 = idx_1;
    float2 _e119 = u_sm.m_0_;
    float2 _e121 = u_sm.m_1_;
    {
        StructWithMat _value2 = ConstructStructWithMat(float2x2(_e75.m_0_, _e75.m_1_));
        {
            float2x2 _value3 = _value2.m;
            s_sm.Store2(0+0, asuint(_value3[0]));
            s_sm.Store2(0+8, asuint(_value3[1]));
        }
    }
    {
        float2x2 _value2 = float2x2(_e82, _e84);
        s_sm.Store2(0+0, asuint(_value2[0]));
        s_sm.Store2(0+8, asuint(_value2[1]));
    }
    s_sm.Store2(0+0, asuint(_e88));
    int _e132 = idx_1;
    s_sm.Store2(_e132*8+0, asuint(float2x2(_e92, _e94)[min(uint(_e89), 1u)]));
    s_sm.Store(0+0+0, asuint(_e100));
    int _e141 = idx_1;
    s_sm.Store(_e141*4+0+0, asuint(_e105));
    int _e145 = idx_1;
    s_sm.Store(0+_e145*8+0, asuint(float2x2(_e109, _e111)[min(uint(_e106), 1u)].x));
    int _e150 = idx_1;
    int _e152 = idx_1;
    s_sm.Store(_e152*4+_e150*8+0, asuint(float2x2(_e119, _e121)[min(uint(_e115), 1u)][min(uint(_e116), 1u)]));
    idx_2 = int(1);
    int _e156 = idx_2;
    idx_2 = asint(asuint(_e156) - asuint(int(1)));
    mat2x2f32_std140_ _e159[4] = u_am;
    v = _e159;
    mat2x2f32_std140_ _e161[4] = v;
    local = Constructarray4_float2x2_(float2x2(_e161[0].c_0_, _e161[0].c_1_), float2x2(_e161[1].c_0_, _e161[1].c_1_), float2x2(_e161[2].c_0_, _e161[2].c_1_), float2x2(_e161[3].c_0_, _e161[3].c_1_));
    float2x2 _e180[4] = local;
    float2 _e184 = u_am[0].c_0_;
    float2 _e186 = u_am[0].c_1_;
    int _e189 = idx_2;
    float2 _e192 = u_am[_e189].c_0_;
    float2 _e194 = u_am[_e189].c_1_;
    float2 _e199 = u_am[0].c_0_;
    int _e202 = idx_2;
    float2 _e204 = u_am[0].c_0_;
    float2 _e206 = u_am[0].c_1_;
    int _e210 = idx_2;
    float2 _e213 = u_am[_e210].c_0_;
    int _e215 = idx_2;
    int _e217 = idx_2;
    float2 _e219 = u_am[_e215].c_0_;
    float2 _e221 = u_am[_e215].c_1_;
    float _e228 = u_am[0].c_0_.x;
    int _e232 = idx_2;
    float _e234 = u_am[0].c_0_[_e232];
    int _e237 = idx_2;
    float2 _e239 = u_am[0].c_0_;
    float2 _e241 = u_am[0].c_1_;
    int _e247 = idx_2;
    int _e248 = idx_2;
    float2 _e250 = u_am[0].c_0_;
    float2 _e252 = u_am[0].c_1_;
    int _e257 = idx_2;
    float _e261 = u_am[_e257].c_0_.x;
    int _e263 = idx_2;
    int _e266 = idx_2;
    float _e268 = u_am[_e263].c_0_[_e266];
    int _e270 = idx_2;
    int _e272 = idx_2;
    float2 _e274 = u_am[_e270].c_0_;
    float2 _e276 = u_am[_e270].c_1_;
    int _e281 = idx_2;
    int _e283 = idx_2;
    int _e284 = idx_2;
    float2 _e286 = u_am[_e281].c_0_;
    float2 _e288 = u_am[_e281].c_1_;
    {
        float2x2 _value2[4] = _e180;
        {
            float2x2 _value3 = _value2[0];
            s_am.Store2(0+0, asuint(_value3[0]));
            s_am.Store2(0+8, asuint(_value3[1]));
        }
        {
            float2x2 _value3 = _value2[1];
            s_am.Store2(16+0, asuint(_value3[0]));
            s_am.Store2(16+8, asuint(_value3[1]));
        }
        {
            float2x2 _value3 = _value2[2];
            s_am.Store2(32+0, asuint(_value3[0]));
            s_am.Store2(32+8, asuint(_value3[1]));
        }
        {
            float2x2 _value3 = _value2[3];
            s_am.Store2(48+0, asuint(_value3[0]));
            s_am.Store2(48+8, asuint(_value3[1]));
        }
    }
    {
        float2x2 _value2 = float2x2(_e184, _e186);
        s_am.Store2(0+0, asuint(_value2[0]));
        s_am.Store2(0+8, asuint(_value2[1]));
    }
    int _e295 = idx_2;
    {
        float2x2 _value2 = float2x2(_e192, _e194);
        s_am.Store2(_e295*16+0, asuint(_value2[0]));
        s_am.Store2(_e295*16+8, asuint(_value2[1]));
    }
    s_am.Store2(0+0, asuint(_e199));
    int _e302 = idx_2;
    s_am.Store2(_e302*8+0, asuint(float2x2(_e204, _e206)[min(uint(_e202), 1u)]));
    int _e305 = idx_2;
    s_am.Store2(0+_e305*16, asuint(_e213));
    int _e309 = idx_2;
    int _e311 = idx_2;
    s_am.Store2(_e311*8+_e309*16, asuint(float2x2(_e219, _e221)[min(uint(_e217), 1u)]));
    s_am.Store(0+0+0, asuint(_e228));
    int _e320 = idx_2;
    s_am.Store(_e320*4+0+0, asuint(_e234));
    int _e324 = idx_2;
    s_am.Store(0+_e324*8+0, asuint(float2x2(_e239, _e241)[min(uint(_e237), 1u)].x));
    int _e329 = idx_2;
    int _e331 = idx_2;
    s_am.Store(_e331*4+_e329*8+0, asuint(float2x2(_e250, _e252)[min(uint(_e247), 1u)][min(uint(_e248), 1u)]));
    int _e334 = idx_2;
    s_am.Store(0+0+_e334*16, asuint(_e261));
    int _e339 = idx_2;
    int _e342 = idx_2;
    s_am.Store(_e342*4+0+_e339*16, asuint(_e268));
    int _e345 = idx_2;
    int _e347 = idx_2;
    s_am.Store(0+_e347*8+_e345*16, asuint(float2x2(_e274, _e276)[min(uint(_e272), 1u)].x));
    int _e351 = idx_2;
    int _e353 = idx_2;
    int _e355 = idx_2;
    s_am.Store(_e355*4+_e353*8+_e351*16, asuint(float2x2(_e286, _e288)[min(uint(_e283), 1u)][min(uint(_e284), 1u)]));
    idx_3 = int(1);
    int _e359 = idx_3;
    idx_3 = asint(asuint(_e359) - asuint(int(1)));
    StructWithArrayOfStructOfMat_std140_ _e362 = u_sasm;
    v_1 = _e362.a;
    StructWithMat_std140_ _e365[4] = v_1;
    local_1 = Constructarray4_StructWithMat_(ConstructStructWithMat(float2x2(_e365[0].m_0_, _e365[0].m_1_)), ConstructStructWithMat(float2x2(_e365[1].m_0_, _e365[1].m_1_)), ConstructStructWithMat(float2x2(_e365[2].m_0_, _e365[2].m_1_)), ConstructStructWithMat(float2x2(_e365[3].m_0_, _e365[3].m_1_)));
    StructWithMat _e388[4] = local_1;
    StructWithMat_std140_ _e392[4] = u_sasm.a;
    v_2 = _e392;
    StructWithMat_std140_ _e394[4] = v_2;
    local_2 = Constructarray4_StructWithMat_(ConstructStructWithMat(float2x2(_e394[0].m_0_, _e394[0].m_1_)), ConstructStructWithMat(float2x2(_e394[1].m_0_, _e394[1].m_1_)), ConstructStructWithMat(float2x2(_e394[2].m_0_, _e394[2].m_1_)), ConstructStructWithMat(float2x2(_e394[3].m_0_, _e394[3].m_1_)));
    StructWithMat _e417[4] = local_2;
    float2 _e422 = u_sasm.a[0].m_0_;
    float2 _e424 = u_sasm.a[0].m_1_;
    int _e428 = idx_3;
    float2 _e431 = u_sasm.a[_e428].m_0_;
    float2 _e433 = u_sasm.a[_e428].m_1_;
    float2 _e439 = u_sasm.a[0].m_0_;
    int _e443 = idx_3;
    float2 _e445 = u_sasm.a[0].m_0_;
    float2 _e447 = u_sasm.a[0].m_1_;
    int _e452 = idx_3;
    float2 _e455 = u_sasm.a[_e452].m_0_;
    int _e458 = idx_3;
    int _e460 = idx_3;
    float2 _e462 = u_sasm.a[_e458].m_0_;
    float2 _e464 = u_sasm.a[_e458].m_1_;
    float _e472 = u_sasm.a[0].m_0_.x;
    int _e477 = idx_3;
    float _e479 = u_sasm.a[0].m_0_[_e477];
    int _e483 = idx_3;
    float2 _e485 = u_sasm.a[0].m_0_;
    float2 _e487 = u_sasm.a[0].m_1_;
    int _e494 = idx_3;
    int _e495 = idx_3;
    float2 _e497 = u_sasm.a[0].m_0_;
    float2 _e499 = u_sasm.a[0].m_1_;
    int _e505 = idx_3;
    float _e509 = u_sasm.a[_e505].m_0_.x;
    int _e512 = idx_3;
    int _e515 = idx_3;
    float _e517 = u_sasm.a[_e512].m_0_[_e515];
    int _e520 = idx_3;
    int _e522 = idx_3;
    float2 _e524 = u_sasm.a[_e520].m_0_;
    float2 _e526 = u_sasm.a[_e520].m_1_;
    int _e532 = idx_3;
    int _e534 = idx_3;
    int _e535 = idx_3;
    float2 _e537 = u_sasm.a[_e532].m_0_;
    float2 _e539 = u_sasm.a[_e532].m_1_;
    {
        StructWithArrayOfStructOfMat _value2 = ConstructStructWithArrayOfStructOfMat(_e388);
        {
            StructWithMat _value3[4] = _value2.a;
            {
                StructWithMat _value4 = _value3[0];
                {
                    float2x2 _value5 = _value4.m;
                    s_sasm.Store2(0+0+0+0, asuint(_value5[0]));
                    s_sasm.Store2(0+0+0+8, asuint(_value5[1]));
                }
            }
            {
                StructWithMat _value4 = _value3[1];
                {
                    float2x2 _value5 = _value4.m;
                    s_sasm.Store2(0+16+0+0, asuint(_value5[0]));
                    s_sasm.Store2(0+16+0+8, asuint(_value5[1]));
                }
            }
            {
                StructWithMat _value4 = _value3[2];
                {
                    float2x2 _value5 = _value4.m;
                    s_sasm.Store2(0+32+0+0, asuint(_value5[0]));
                    s_sasm.Store2(0+32+0+8, asuint(_value5[1]));
                }
            }
            {
                StructWithMat _value4 = _value3[3];
                {
                    float2x2 _value5 = _value4.m;
                    s_sasm.Store2(0+48+0+0, asuint(_value5[0]));
                    s_sasm.Store2(0+48+0+8, asuint(_value5[1]));
                }
            }
        }
    }
    {
        StructWithMat _value2[4] = _e417;
        {
            StructWithMat _value3 = _value2[0];
            {
                float2x2 _value4 = _value3.m;
                s_sasm.Store2(0+0+0+0, asuint(_value4[0]));
                s_sasm.Store2(0+0+0+8, asuint(_value4[1]));
            }
        }
        {
            StructWithMat _value3 = _value2[1];
            {
                float2x2 _value4 = _value3.m;
                s_sasm.Store2(0+16+0+0, asuint(_value4[0]));
                s_sasm.Store2(0+16+0+8, asuint(_value4[1]));
            }
        }
        {
            StructWithMat _value3 = _value2[2];
            {
                float2x2 _value4 = _value3.m;
                s_sasm.Store2(0+32+0+0, asuint(_value4[0]));
                s_sasm.Store2(0+32+0+8, asuint(_value4[1]));
            }
        }
        {
            StructWithMat _value3 = _value2[3];
            {
                float2x2 _value4 = _value3.m;
                s_sasm.Store2(0+48+0+0, asuint(_value4[0]));
                s_sasm.Store2(0+48+0+8, asuint(_value4[1]));
            }
        }
    }
    {
        float2x2 _value2 = float2x2(_e422, _e424);
        s_sasm.Store2(0+0+0+0, asuint(_value2[0]));
        s_sasm.Store2(0+0+0+8, asuint(_value2[1]));
    }
    int _e551 = idx_3;
    {
        float2x2 _value2 = float2x2(_e431, _e433);
        s_sasm.Store2(0+_e551*16+0+0, asuint(_value2[0]));
        s_sasm.Store2(0+_e551*16+0+8, asuint(_value2[1]));
    }
    s_sasm.Store2(0+0+0+0, asuint(_e439));
    int _e563 = idx_3;
    s_sasm.Store2(_e563*8+0+0+0, asuint(float2x2(_e445, _e447)[min(uint(_e443), 1u)]));
    int _e567 = idx_3;
    s_sasm.Store2(0+0+_e567*16+0, asuint(_e455));
    int _e573 = idx_3;
    int _e576 = idx_3;
    s_sasm.Store2(_e576*8+0+_e573*16+0, asuint(float2x2(_e462, _e464)[min(uint(_e460), 1u)]));
    s_sasm.Store(0+0+0+0+0, asuint(_e472));
    int _e589 = idx_3;
    s_sasm.Store(_e589*4+0+0+0+0, asuint(_e479));
    int _e595 = idx_3;
    s_sasm.Store(0+_e595*8+0+0+0, asuint(float2x2(_e485, _e487)[min(uint(_e483), 1u)].x));
    int _e602 = idx_3;
    int _e604 = idx_3;
    s_sasm.Store(_e604*4+_e602*8+0+0+0, asuint(float2x2(_e497, _e499)[min(uint(_e494), 1u)][min(uint(_e495), 1u)]));
    int _e608 = idx_3;
    s_sasm.Store(0+0+0+_e608*16+0, asuint(_e509));
    int _e615 = idx_3;
    int _e619 = idx_3;
    s_sasm.Store(_e619*4+0+0+_e615*16+0, asuint(_e517));
    int _e623 = idx_3;
    int _e626 = idx_3;
    s_sasm.Store(0+_e626*8+0+_e623*16+0, asuint(float2x2(_e524, _e526)[min(uint(_e522), 1u)].x));
    int _e631 = idx_3;
    int _e634 = idx_3;
    int _e636 = idx_3;
    s_sasm.Store(_e636*4+_e634*8+0+_e631*16+0, asuint(float2x2(_e537, _e539)[min(uint(_e534), 1u)][min(uint(_e535), 1u)]));
    return;
}
