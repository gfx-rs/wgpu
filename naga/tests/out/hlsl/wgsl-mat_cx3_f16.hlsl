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

void access_m()
{
    int idx = int(1);

    int _e2 = idx;
    idx = asint(asuint(_e2) - asuint(int(1)));
    half4x3 l_s_m = half4x3(s_m.Load<half3>(0), s_m.Load<half3>(8), s_m.Load<half3>(16), s_m.Load<half3>(24));
    half3 l_s_c_c = s_m.Load<half3>(0);
    int _e10 = idx;
    half3 l_s_c_v = s_m.Load<half3>(_e10*8);
    half l_s_e_cc = s_m.Load<half>(0+0);
    int _e19 = idx;
    half l_s_e_cv = s_m.Load<half>(_e19*2+0);
    int _e23 = idx;
    half l_s_e_vc = s_m.Load<half>(0+_e23*8);
    int _e28 = idx;
    int _e30 = idx;
    half l_s_e_vv = s_m.Load<half>(_e30*2+_e28*8);
    half3 _e35 = u_m.c_0_;
    half3 _e37 = u_m.c_1_;
    half3 _e39 = u_m.c_2_;
    half3 _e41 = u_m.c_3_;
    half4x3 l_u_m = half4x3(_e35, _e37, _e39, _e41);
    half3 l_u_c_c = u_m.c_0_;
    int _e46 = idx;
    half3 _e49 = u_m.c_0_;
    half3 _e51 = u_m.c_1_;
    half3 _e53 = u_m.c_2_;
    half3 _e55 = u_m.c_3_;
    half3 l_u_c_v = half4x3(_e49, _e51, _e53, _e55)[min(uint(_e46), 3u)];
    half l_u_e_cc = u_m.c_0_.x;
    int _e64 = idx;
    half l_u_e_cv = u_m.c_0_[_e64];
    int _e67 = idx;
    half3 _e70 = u_m.c_0_;
    half3 _e72 = u_m.c_1_;
    half3 _e74 = u_m.c_2_;
    half3 _e76 = u_m.c_3_;
    half l_u_e_vc = half4x3(_e70, _e72, _e74, _e76)[min(uint(_e67), 3u)].x;
    int _e80 = idx;
    int _e81 = idx;
    half3 _e84 = u_m.c_0_;
    half3 _e86 = u_m.c_1_;
    half3 _e88 = u_m.c_2_;
    half3 _e90 = u_m.c_3_;
    half l_u_e_vv = half4x3(_e84, _e86, _e88, _e90)[min(uint(_e80), 3u)][min(uint(_e81), 2u)];
    {
        half4x3 _value2 = l_u_m;
        s_m.Store(0, _value2[0]);
        s_m.Store(8, _value2[1]);
        s_m.Store(16, _value2[2]);
        s_m.Store(24, _value2[3]);
    }
    s_m.Store(0, l_u_c_c);
    int _e97 = idx;
    s_m.Store(_e97*8, l_u_c_v);
    s_m.Store(0+0, l_u_e_cc);
    int _e104 = idx;
    s_m.Store(_e104*2+0, l_u_e_cv);
    int _e107 = idx;
    s_m.Store(0+_e107*8, l_u_e_vc);
    int _e111 = idx;
    int _e113 = idx;
    s_m.Store(_e113*2+_e111*8, l_u_e_vv);
    return;
}

StructWithMat ConstructStructWithMat(half4x3 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m = arg0;
    return ret;
}

void access_sm()
{
    int idx_1 = int(1);

    int _e2 = idx_1;
    idx_1 = asint(asuint(_e2) - asuint(int(1)));
    StructWithMat l_s_s = ConstructStructWithMat(half4x3(s_sm.Load<half3>(0+0), s_sm.Load<half3>(0+8), s_sm.Load<half3>(0+16), s_sm.Load<half3>(0+24)));
    half4x3 l_s_m_1 = half4x3(s_sm.Load<half3>(0+0), s_sm.Load<half3>(0+8), s_sm.Load<half3>(0+16), s_sm.Load<half3>(0+24));
    half3 l_s_c_c_1 = s_sm.Load<half3>(0+0);
    int _e15 = idx_1;
    half3 l_s_c_v_1 = s_sm.Load<half3>(_e15*8+0);
    half l_s_e_cc_1 = s_sm.Load<half>(0+0+0);
    int _e26 = idx_1;
    half l_s_e_cv_1 = s_sm.Load<half>(_e26*2+0+0);
    int _e31 = idx_1;
    half l_s_e_vc_1 = s_sm.Load<half>(0+_e31*8+0);
    int _e37 = idx_1;
    int _e39 = idx_1;
    half l_s_e_vv_1 = s_sm.Load<half>(_e39*2+_e37*8+0);
    StructWithMat_std140_ _e43 = u_sm;
    StructWithMat l_u_s = ConstructStructWithMat(half4x3(_e43.m_0_, _e43.m_1_, _e43.m_2_, _e43.m_3_));
    half3 _e52 = u_sm.m_0_;
    half3 _e54 = u_sm.m_1_;
    half3 _e56 = u_sm.m_2_;
    half3 _e58 = u_sm.m_3_;
    half4x3 l_u_m_1 = half4x3(_e52, _e54, _e56, _e58);
    half3 l_u_c_c_1 = u_sm.m_0_;
    int _e63 = idx_1;
    half3 _e66 = u_sm.m_0_;
    half3 _e68 = u_sm.m_1_;
    half3 _e70 = u_sm.m_2_;
    half3 _e72 = u_sm.m_3_;
    half3 l_u_c_v_1 = half4x3(_e66, _e68, _e70, _e72)[min(uint(_e63), 3u)];
    half l_u_e_cc_1 = u_sm.m_0_.x;
    int _e81 = idx_1;
    half l_u_e_cv_1 = u_sm.m_0_[_e81];
    int _e84 = idx_1;
    half3 _e87 = u_sm.m_0_;
    half3 _e89 = u_sm.m_1_;
    half3 _e91 = u_sm.m_2_;
    half3 _e93 = u_sm.m_3_;
    half l_u_e_vc_1 = half4x3(_e87, _e89, _e91, _e93)[min(uint(_e84), 3u)].x;
    int _e97 = idx_1;
    int _e98 = idx_1;
    half3 _e101 = u_sm.m_0_;
    half3 _e103 = u_sm.m_1_;
    half3 _e105 = u_sm.m_2_;
    half3 _e107 = u_sm.m_3_;
    half l_u_e_vv_1 = half4x3(_e101, _e103, _e105, _e107)[min(uint(_e97), 3u)][min(uint(_e98), 2u)];
    {
        StructWithMat _value2 = l_u_s;
        {
            half4x3 _value3 = _value2.m;
            s_sm.Store(0+0, _value3[0]);
            s_sm.Store(0+8, _value3[1]);
            s_sm.Store(0+16, _value3[2]);
            s_sm.Store(0+24, _value3[3]);
        }
    }
    {
        half4x3 _value2 = l_u_m_1;
        s_sm.Store(0+0, _value2[0]);
        s_sm.Store(0+8, _value2[1]);
        s_sm.Store(0+16, _value2[2]);
        s_sm.Store(0+24, _value2[3]);
    }
    s_sm.Store(0+0, l_u_c_c_1);
    int _e118 = idx_1;
    s_sm.Store(_e118*8+0, l_u_c_v_1);
    s_sm.Store(0+0+0, l_u_e_cc_1);
    int _e127 = idx_1;
    s_sm.Store(_e127*2+0+0, l_u_e_cv_1);
    int _e131 = idx_1;
    s_sm.Store(0+_e131*8+0, l_u_e_vc_1);
    int _e136 = idx_1;
    int _e138 = idx_1;
    s_sm.Store(_e138*2+_e136*8+0, l_u_e_vv_1);
    return;
}

typedef half4x3 ret_Constructarray4_half4x3_[4];
ret_Constructarray4_half4x3_ Constructarray4_half4x3_(half4x3 arg0, half4x3 arg1, half4x3 arg2, half4x3 arg3) {
    half4x3 ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef half4x3 ret_convert_std140_0_[4];
ret_convert_std140_0_ convert_std140_0_(mat4x3f16_std140_ v[4])
{
    return Constructarray4_half4x3_(half4x3(v[0].c_0_, v[0].c_1_, v[0].c_2_, v[0].c_3_), half4x3(v[1].c_0_, v[1].c_1_, v[1].c_2_, v[1].c_3_), half4x3(v[2].c_0_, v[2].c_1_, v[2].c_2_, v[2].c_3_), half4x3(v[3].c_0_, v[3].c_1_, v[3].c_2_, v[3].c_3_));
}

void access_am()
{
    int idx_2 = int(1);

    int _e2 = idx_2;
    idx_2 = asint(asuint(_e2) - asuint(int(1)));
    half4x3 l_s_a[4] = Constructarray4_half4x3_(half4x3(s_am.Load<half3>(0+0), s_am.Load<half3>(0+8), s_am.Load<half3>(0+16), s_am.Load<half3>(0+24)), half4x3(s_am.Load<half3>(32+0), s_am.Load<half3>(32+8), s_am.Load<half3>(32+16), s_am.Load<half3>(32+24)), half4x3(s_am.Load<half3>(64+0), s_am.Load<half3>(64+8), s_am.Load<half3>(64+16), s_am.Load<half3>(64+24)), half4x3(s_am.Load<half3>(96+0), s_am.Load<half3>(96+8), s_am.Load<half3>(96+16), s_am.Load<half3>(96+24)));
    half4x3 l_s_m_c = half4x3(s_am.Load<half3>(0+0), s_am.Load<half3>(0+8), s_am.Load<half3>(0+16), s_am.Load<half3>(0+24));
    int _e10 = idx_2;
    half4x3 l_s_m_v = half4x3(s_am.Load<half3>(_e10*32+0), s_am.Load<half3>(_e10*32+8), s_am.Load<half3>(_e10*32+16), s_am.Load<half3>(_e10*32+24));
    half3 l_s_c_cc = s_am.Load<half3>(0+0);
    int _e19 = idx_2;
    half3 l_s_c_cv = s_am.Load<half3>(_e19*8+0);
    int _e23 = idx_2;
    half3 l_s_c_vc = s_am.Load<half3>(0+_e23*32);
    int _e28 = idx_2;
    int _e30 = idx_2;
    half3 l_s_c_vv = s_am.Load<half3>(_e30*8+_e28*32);
    half l_s_e_ccc = s_am.Load<half>(0+0+0);
    int _e41 = idx_2;
    half l_s_e_ccv = s_am.Load<half>(_e41*2+0+0);
    int _e46 = idx_2;
    half l_s_e_cvc = s_am.Load<half>(0+_e46*8+0);
    int _e52 = idx_2;
    int _e54 = idx_2;
    half l_s_e_cvv = s_am.Load<half>(_e54*2+_e52*8+0);
    int _e58 = idx_2;
    half l_s_e_vcc = s_am.Load<half>(0+0+_e58*32);
    int _e64 = idx_2;
    int _e67 = idx_2;
    half l_s_e_vcv = s_am.Load<half>(_e67*2+0+_e64*32);
    int _e71 = idx_2;
    int _e73 = idx_2;
    half l_s_e_vvc = s_am.Load<half>(0+_e73*8+_e71*32);
    int _e78 = idx_2;
    int _e80 = idx_2;
    int _e82 = idx_2;
    half l_s_e_vvv = s_am.Load<half>(_e82*2+_e80*8+_e78*32);
    mat4x3f16_std140_ _e86[4] = u_am;
    const half4x3 _e296[4] = convert_std140_0_(_e86);
    half3 _e90 = u_am[0].c_0_;
    half3 _e92 = u_am[0].c_1_;
    half3 _e94 = u_am[0].c_2_;
    half3 _e96 = u_am[0].c_3_;
    half4x3 l_u_m_c = half4x3(_e90, _e92, _e94, _e96);
    int _e99 = idx_2;
    half3 _e102 = u_am[_e99].c_0_;
    half3 _e104 = u_am[_e99].c_1_;
    half3 _e106 = u_am[_e99].c_2_;
    half3 _e108 = u_am[_e99].c_3_;
    half4x3 l_u_m_v = half4x3(_e102, _e104, _e106, _e108);
    half3 l_u_c_cc = u_am[0].c_0_;
    int _e116 = idx_2;
    half3 _e118 = u_am[0].c_0_;
    half3 _e120 = u_am[0].c_1_;
    half3 _e122 = u_am[0].c_2_;
    half3 _e124 = u_am[0].c_3_;
    half3 l_u_c_cv = half4x3(_e118, _e120, _e122, _e124)[min(uint(_e116), 3u)];
    int _e128 = idx_2;
    half3 l_u_c_vc = u_am[_e128].c_0_;
    int _e133 = idx_2;
    int _e135 = idx_2;
    half3 _e137 = u_am[_e133].c_0_;
    half3 _e139 = u_am[_e133].c_1_;
    half3 _e141 = u_am[_e133].c_2_;
    half3 _e143 = u_am[_e133].c_3_;
    half3 l_u_c_vv = half4x3(_e137, _e139, _e141, _e143)[min(uint(_e135), 3u)];
    half l_u_e_ccc = u_am[0].c_0_.x;
    int _e154 = idx_2;
    half l_u_e_ccv = u_am[0].c_0_[_e154];
    int _e159 = idx_2;
    half3 _e161 = u_am[0].c_0_;
    half3 _e163 = u_am[0].c_1_;
    half3 _e165 = u_am[0].c_2_;
    half3 _e167 = u_am[0].c_3_;
    half l_u_e_cvc = half4x3(_e161, _e163, _e165, _e167)[min(uint(_e159), 3u)].x;
    int _e173 = idx_2;
    int _e174 = idx_2;
    half3 _e176 = u_am[0].c_0_;
    half3 _e178 = u_am[0].c_1_;
    half3 _e180 = u_am[0].c_2_;
    half3 _e182 = u_am[0].c_3_;
    half l_u_e_cvv = half4x3(_e176, _e178, _e180, _e182)[min(uint(_e173), 3u)][min(uint(_e174), 2u)];
    int _e187 = idx_2;
    half l_u_e_vcc = u_am[_e187].c_0_.x;
    int _e193 = idx_2;
    int _e196 = idx_2;
    half l_u_e_vcv = u_am[_e193].c_0_[_e196];
    int _e200 = idx_2;
    int _e202 = idx_2;
    half3 _e204 = u_am[_e200].c_0_;
    half3 _e206 = u_am[_e200].c_1_;
    half3 _e208 = u_am[_e200].c_2_;
    half3 _e210 = u_am[_e200].c_3_;
    half l_u_e_vvc = half4x3(_e204, _e206, _e208, _e210)[min(uint(_e202), 3u)].x;
    int _e215 = idx_2;
    int _e217 = idx_2;
    int _e218 = idx_2;
    half3 _e220 = u_am[_e215].c_0_;
    half3 _e222 = u_am[_e215].c_1_;
    half3 _e224 = u_am[_e215].c_2_;
    half3 _e226 = u_am[_e215].c_3_;
    half l_u_e_vvv = half4x3(_e220, _e222, _e224, _e226)[min(uint(_e217), 3u)][min(uint(_e218), 2u)];
    {
        half4x3 _value2[4] = _e296;
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
        half4x3 _value2 = l_u_m_c;
        s_am.Store(0+0, _value2[0]);
        s_am.Store(0+8, _value2[1]);
        s_am.Store(0+16, _value2[2]);
        s_am.Store(0+24, _value2[3]);
    }
    int _e233 = idx_2;
    {
        half4x3 _value2 = l_u_m_v;
        s_am.Store(_e233*32+0, _value2[0]);
        s_am.Store(_e233*32+8, _value2[1]);
        s_am.Store(_e233*32+16, _value2[2]);
        s_am.Store(_e233*32+24, _value2[3]);
    }
    s_am.Store(0+0, l_u_c_cc);
    int _e240 = idx_2;
    s_am.Store(_e240*8+0, l_u_c_cv);
    int _e243 = idx_2;
    s_am.Store(0+_e243*32, l_u_c_vc);
    int _e247 = idx_2;
    int _e249 = idx_2;
    s_am.Store(_e249*8+_e247*32, l_u_c_vv);
    s_am.Store(0+0+0, l_u_e_ccc);
    int _e258 = idx_2;
    s_am.Store(_e258*2+0+0, l_u_e_ccv);
    int _e262 = idx_2;
    s_am.Store(0+_e262*8+0, l_u_e_cvc);
    int _e267 = idx_2;
    int _e269 = idx_2;
    s_am.Store(_e269*2+_e267*8+0, l_u_e_cvv);
    int _e272 = idx_2;
    s_am.Store(0+0+_e272*32, l_u_e_vcc);
    int _e277 = idx_2;
    int _e280 = idx_2;
    s_am.Store(_e280*2+0+_e277*32, l_u_e_vcv);
    int _e283 = idx_2;
    int _e285 = idx_2;
    s_am.Store(0+_e285*8+_e283*32, l_u_e_vvc);
    int _e289 = idx_2;
    int _e291 = idx_2;
    int _e293 = idx_2;
    s_am.Store(_e293*2+_e291*8+_e289*32, l_u_e_vvv);
    return;
}

typedef StructWithMat ret_Constructarray4_StructWithMat_[4];
ret_Constructarray4_StructWithMat_ Constructarray4_StructWithMat_(StructWithMat arg0, StructWithMat arg1, StructWithMat arg2, StructWithMat arg3) {
    StructWithMat ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef StructWithMat ret_convert_std140_1_[4];
ret_convert_std140_1_ convert_std140_1_(StructWithMat_std140_ v_1[4])
{
    return Constructarray4_StructWithMat_(ConstructStructWithMat(half4x3(v_1[0].m_0_, v_1[0].m_1_, v_1[0].m_2_, v_1[0].m_3_)), ConstructStructWithMat(half4x3(v_1[1].m_0_, v_1[1].m_1_, v_1[1].m_2_, v_1[1].m_3_)), ConstructStructWithMat(half4x3(v_1[2].m_0_, v_1[2].m_1_, v_1[2].m_2_, v_1[2].m_3_)), ConstructStructWithMat(half4x3(v_1[3].m_0_, v_1[3].m_1_, v_1[3].m_2_, v_1[3].m_3_)));
}

StructWithArrayOfStructOfMat ConstructStructWithArrayOfStructOfMat(StructWithMat arg0[4]) {
    StructWithArrayOfStructOfMat ret = (StructWithArrayOfStructOfMat)0;
    ret.a = arg0;
    return ret;
}

void access_sasm()
{
    int idx_3 = int(1);

    int _e2 = idx_3;
    idx_3 = asint(asuint(_e2) - asuint(int(1)));
    StructWithArrayOfStructOfMat l_s_s_1 = ConstructStructWithArrayOfStructOfMat(Constructarray4_StructWithMat_(ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+0+0+0), s_sasm.Load<half3>(0+0+0+8), s_sasm.Load<half3>(0+0+0+16), s_sasm.Load<half3>(0+0+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+32+0+0), s_sasm.Load<half3>(0+32+0+8), s_sasm.Load<half3>(0+32+0+16), s_sasm.Load<half3>(0+32+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+64+0+0), s_sasm.Load<half3>(0+64+0+8), s_sasm.Load<half3>(0+64+0+16), s_sasm.Load<half3>(0+64+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+96+0+0), s_sasm.Load<half3>(0+96+0+8), s_sasm.Load<half3>(0+96+0+16), s_sasm.Load<half3>(0+96+0+24)))));
    StructWithMat l_s_a_1[4] = Constructarray4_StructWithMat_(ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+0+0+0), s_sasm.Load<half3>(0+0+0+8), s_sasm.Load<half3>(0+0+0+16), s_sasm.Load<half3>(0+0+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+32+0+0), s_sasm.Load<half3>(0+32+0+8), s_sasm.Load<half3>(0+32+0+16), s_sasm.Load<half3>(0+32+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+64+0+0), s_sasm.Load<half3>(0+64+0+8), s_sasm.Load<half3>(0+64+0+16), s_sasm.Load<half3>(0+64+0+24))), ConstructStructWithMat(half4x3(s_sasm.Load<half3>(0+96+0+0), s_sasm.Load<half3>(0+96+0+8), s_sasm.Load<half3>(0+96+0+16), s_sasm.Load<half3>(0+96+0+24))));
    half4x3 l_s_m_c_1 = half4x3(s_sasm.Load<half3>(0+0+0+0), s_sasm.Load<half3>(0+0+0+8), s_sasm.Load<half3>(0+0+0+16), s_sasm.Load<half3>(0+0+0+24));
    int _e16 = idx_3;
    half4x3 l_s_m_v_1 = half4x3(s_sasm.Load<half3>(0+_e16*32+0+0), s_sasm.Load<half3>(0+_e16*32+0+8), s_sasm.Load<half3>(0+_e16*32+0+16), s_sasm.Load<half3>(0+_e16*32+0+24));
    half3 l_s_c_cc_1 = s_sasm.Load<half3>(0+0+0+0);
    int _e30 = idx_3;
    half3 l_s_c_cv_1 = s_sasm.Load<half3>(_e30*8+0+0+0);
    int _e35 = idx_3;
    half3 l_s_c_vc_1 = s_sasm.Load<half3>(0+0+_e35*32+0);
    int _e42 = idx_3;
    int _e45 = idx_3;
    half3 l_s_c_vv_1 = s_sasm.Load<half3>(_e45*8+0+_e42*32+0);
    half l_s_e_ccc_1 = s_sasm.Load<half>(0+0+0+0+0);
    int _e60 = idx_3;
    half l_s_e_ccv_1 = s_sasm.Load<half>(_e60*2+0+0+0+0);
    int _e67 = idx_3;
    half l_s_e_cvc_1 = s_sasm.Load<half>(0+_e67*8+0+0+0);
    int _e75 = idx_3;
    int _e77 = idx_3;
    half l_s_e_cvv_1 = s_sasm.Load<half>(_e77*2+_e75*8+0+0+0);
    int _e82 = idx_3;
    half l_s_e_vcc_1 = s_sasm.Load<half>(0+0+0+_e82*32+0);
    int _e90 = idx_3;
    int _e94 = idx_3;
    half l_s_e_vcv_1 = s_sasm.Load<half>(_e94*2+0+0+_e90*32+0);
    int _e99 = idx_3;
    int _e102 = idx_3;
    half l_s_e_vvc_1 = s_sasm.Load<half>(0+_e102*8+0+_e99*32+0);
    int _e108 = idx_3;
    int _e111 = idx_3;
    int _e113 = idx_3;
    half l_s_e_vvv_1 = s_sasm.Load<half>(_e113*2+_e111*8+0+_e108*32+0);
    StructWithArrayOfStructOfMat_std140_ _e117 = u_sasm;
    const StructWithMat _e119[4] = convert_std140_1_(_e117.a);
    StructWithArrayOfStructOfMat l_u_s_1 = ConstructStructWithArrayOfStructOfMat(_e119);
    StructWithMat_std140_ _e123[4] = u_sasm.a;
    const StructWithMat _e377[4] = convert_std140_1_(_e123);
    half3 _e128 = u_sasm.a[0].m_0_;
    half3 _e130 = u_sasm.a[0].m_1_;
    half3 _e132 = u_sasm.a[0].m_2_;
    half3 _e134 = u_sasm.a[0].m_3_;
    half4x3 l_u_m_c_1 = half4x3(_e128, _e130, _e132, _e134);
    int _e138 = idx_3;
    half3 _e141 = u_sasm.a[_e138].m_0_;
    half3 _e143 = u_sasm.a[_e138].m_1_;
    half3 _e145 = u_sasm.a[_e138].m_2_;
    half3 _e147 = u_sasm.a[_e138].m_3_;
    half4x3 l_u_m_v_1 = half4x3(_e141, _e143, _e145, _e147);
    half3 l_u_c_cc_1 = u_sasm.a[0].m_0_;
    int _e157 = idx_3;
    half3 _e159 = u_sasm.a[0].m_0_;
    half3 _e161 = u_sasm.a[0].m_1_;
    half3 _e163 = u_sasm.a[0].m_2_;
    half3 _e165 = u_sasm.a[0].m_3_;
    half3 l_u_c_cv_1 = half4x3(_e159, _e161, _e163, _e165)[min(uint(_e157), 3u)];
    int _e170 = idx_3;
    half3 l_u_c_vc_1 = u_sasm.a[_e170].m_0_;
    int _e176 = idx_3;
    int _e178 = idx_3;
    half3 _e180 = u_sasm.a[_e176].m_0_;
    half3 _e182 = u_sasm.a[_e176].m_1_;
    half3 _e184 = u_sasm.a[_e176].m_2_;
    half3 _e186 = u_sasm.a[_e176].m_3_;
    half3 l_u_c_vv_1 = half4x3(_e180, _e182, _e184, _e186)[min(uint(_e178), 3u)];
    half l_u_e_ccc_1 = u_sasm.a[0].m_0_.x;
    int _e199 = idx_3;
    half l_u_e_ccv_1 = u_sasm.a[0].m_0_[_e199];
    int _e205 = idx_3;
    half3 _e207 = u_sasm.a[0].m_0_;
    half3 _e209 = u_sasm.a[0].m_1_;
    half3 _e211 = u_sasm.a[0].m_2_;
    half3 _e213 = u_sasm.a[0].m_3_;
    half l_u_e_cvc_1 = half4x3(_e207, _e209, _e211, _e213)[min(uint(_e205), 3u)].x;
    int _e220 = idx_3;
    int _e221 = idx_3;
    half3 _e223 = u_sasm.a[0].m_0_;
    half3 _e225 = u_sasm.a[0].m_1_;
    half3 _e227 = u_sasm.a[0].m_2_;
    half3 _e229 = u_sasm.a[0].m_3_;
    half l_u_e_cvv_1 = half4x3(_e223, _e225, _e227, _e229)[min(uint(_e220), 3u)][min(uint(_e221), 2u)];
    int _e235 = idx_3;
    half l_u_e_vcc_1 = u_sasm.a[_e235].m_0_.x;
    int _e242 = idx_3;
    int _e245 = idx_3;
    half l_u_e_vcv_1 = u_sasm.a[_e242].m_0_[_e245];
    int _e250 = idx_3;
    int _e252 = idx_3;
    half3 _e254 = u_sasm.a[_e250].m_0_;
    half3 _e256 = u_sasm.a[_e250].m_1_;
    half3 _e258 = u_sasm.a[_e250].m_2_;
    half3 _e260 = u_sasm.a[_e250].m_3_;
    half l_u_e_vvc_1 = half4x3(_e254, _e256, _e258, _e260)[min(uint(_e252), 3u)].x;
    int _e266 = idx_3;
    int _e268 = idx_3;
    int _e269 = idx_3;
    half3 _e271 = u_sasm.a[_e266].m_0_;
    half3 _e273 = u_sasm.a[_e266].m_1_;
    half3 _e275 = u_sasm.a[_e266].m_2_;
    half3 _e277 = u_sasm.a[_e266].m_3_;
    half l_u_e_vvv_1 = half4x3(_e271, _e273, _e275, _e277)[min(uint(_e268), 3u)][min(uint(_e269), 2u)];
    {
        StructWithArrayOfStructOfMat _value2 = l_u_s_1;
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
        StructWithMat _value2[4] = _e377;
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
        half4x3 _value2 = l_u_m_c_1;
        s_sasm.Store(0+0+0+0, _value2[0]);
        s_sasm.Store(0+0+0+8, _value2[1]);
        s_sasm.Store(0+0+0+16, _value2[2]);
        s_sasm.Store(0+0+0+24, _value2[3]);
    }
    int _e289 = idx_3;
    {
        half4x3 _value2 = l_u_m_v_1;
        s_sasm.Store(0+_e289*32+0+0, _value2[0]);
        s_sasm.Store(0+_e289*32+0+8, _value2[1]);
        s_sasm.Store(0+_e289*32+0+16, _value2[2]);
        s_sasm.Store(0+_e289*32+0+24, _value2[3]);
    }
    s_sasm.Store(0+0+0+0, l_u_c_cc_1);
    int _e301 = idx_3;
    s_sasm.Store(_e301*8+0+0+0, l_u_c_cv_1);
    int _e305 = idx_3;
    s_sasm.Store(0+0+_e305*32+0, l_u_c_vc_1);
    int _e311 = idx_3;
    int _e314 = idx_3;
    s_sasm.Store(_e314*8+0+_e311*32+0, l_u_c_vv_1);
    s_sasm.Store(0+0+0+0+0, l_u_e_ccc_1);
    int _e327 = idx_3;
    s_sasm.Store(_e327*2+0+0+0+0, l_u_e_ccv_1);
    int _e333 = idx_3;
    s_sasm.Store(0+_e333*8+0+0+0, l_u_e_cvc_1);
    int _e340 = idx_3;
    int _e342 = idx_3;
    s_sasm.Store(_e342*2+_e340*8+0+0+0, l_u_e_cvv_1);
    int _e346 = idx_3;
    s_sasm.Store(0+0+0+_e346*32+0, l_u_e_vcc_1);
    int _e353 = idx_3;
    int _e357 = idx_3;
    s_sasm.Store(_e357*2+0+0+_e353*32+0, l_u_e_vcv_1);
    int _e361 = idx_3;
    int _e364 = idx_3;
    s_sasm.Store(0+_e364*8+0+_e361*32+0, l_u_e_vvc_1);
    int _e369 = idx_3;
    int _e372 = idx_3;
    int _e374 = idx_3;
    s_sasm.Store(_e374*2+_e372*8+0+_e369*32+0, l_u_e_vvv_1);
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    access_m();
    access_sm();
    access_am();
    access_sasm();
    return;
}
