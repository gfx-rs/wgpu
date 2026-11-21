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

void access_m()
{
    int idx = int(1);

    int _e2 = idx;
    idx = asint(asuint(_e2) - asuint(int(1)));
    float2x2 l_s_m = float2x2(asfloat(s_m.Load2(0)), asfloat(s_m.Load2(8)));
    float2 l_s_c_c = asfloat(s_m.Load2(0));
    int _e10 = idx;
    float2 l_s_c_v = asfloat(s_m.Load2(_e10*8));
    float l_s_e_cc = asfloat(s_m.Load(0+0));
    int _e19 = idx;
    float l_s_e_cv = asfloat(s_m.Load(_e19*4+0));
    int _e23 = idx;
    float l_s_e_vc = asfloat(s_m.Load(0+_e23*8));
    int _e28 = idx;
    int _e30 = idx;
    float l_s_e_vv = asfloat(s_m.Load(_e30*4+_e28*8));
    float2 _e35 = u_m.c_0_;
    float2 _e37 = u_m.c_1_;
    float2x2 l_u_m = float2x2(_e35, _e37);
    float2 l_u_c_c = u_m.c_0_;
    int _e42 = idx;
    float2 _e45 = u_m.c_0_;
    float2 _e47 = u_m.c_1_;
    float2 l_u_c_v = float2x2(_e45, _e47)[min(uint(_e42), 1u)];
    float l_u_e_cc = u_m.c_0_.x;
    int _e56 = idx;
    float l_u_e_cv = u_m.c_0_[_e56];
    int _e59 = idx;
    float2 _e62 = u_m.c_0_;
    float2 _e64 = u_m.c_1_;
    float l_u_e_vc = float2x2(_e62, _e64)[min(uint(_e59), 1u)].x;
    int _e68 = idx;
    int _e69 = idx;
    float2 _e72 = u_m.c_0_;
    float2 _e74 = u_m.c_1_;
    float l_u_e_vv = float2x2(_e72, _e74)[min(uint(_e68), 1u)][min(uint(_e69), 1u)];
    {
        float2x2 _value2 = l_u_m;
        s_m.Store2(0, asuint(_value2[0]));
        s_m.Store2(8, asuint(_value2[1]));
    }
    s_m.Store2(0, asuint(l_u_c_c));
    int _e81 = idx;
    s_m.Store2(_e81*8, asuint(l_u_c_v));
    s_m.Store(0+0, asuint(l_u_e_cc));
    int _e88 = idx;
    s_m.Store(_e88*4+0, asuint(l_u_e_cv));
    int _e91 = idx;
    s_m.Store(0+_e91*8, asuint(l_u_e_vc));
    int _e95 = idx;
    int _e97 = idx;
    s_m.Store(_e97*4+_e95*8, asuint(l_u_e_vv));
    return;
}

StructWithMat ConstructStructWithMat(float2x2 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m = arg0;
    return ret;
}

void access_sm()
{
    int idx_1 = int(1);

    int _e2 = idx_1;
    idx_1 = asint(asuint(_e2) - asuint(int(1)));
    StructWithMat l_s_s = ConstructStructWithMat(float2x2(asfloat(s_sm.Load2(0+0)), asfloat(s_sm.Load2(0+8))));
    float2x2 l_s_m_1 = float2x2(asfloat(s_sm.Load2(0+0)), asfloat(s_sm.Load2(0+8)));
    float2 l_s_c_c_1 = asfloat(s_sm.Load2(0+0));
    int _e15 = idx_1;
    float2 l_s_c_v_1 = asfloat(s_sm.Load2(_e15*8+0));
    float l_s_e_cc_1 = asfloat(s_sm.Load(0+0+0));
    int _e26 = idx_1;
    float l_s_e_cv_1 = asfloat(s_sm.Load(_e26*4+0+0));
    int _e31 = idx_1;
    float l_s_e_vc_1 = asfloat(s_sm.Load(0+_e31*8+0));
    int _e37 = idx_1;
    int _e39 = idx_1;
    float l_s_e_vv_1 = asfloat(s_sm.Load(_e39*4+_e37*8+0));
    StructWithMat_std140_ _e43 = u_sm;
    StructWithMat l_u_s = ConstructStructWithMat(float2x2(_e43.m_0_, _e43.m_1_));
    float2 _e50 = u_sm.m_0_;
    float2 _e52 = u_sm.m_1_;
    float2x2 l_u_m_1 = float2x2(_e50, _e52);
    float2 l_u_c_c_1 = u_sm.m_0_;
    int _e57 = idx_1;
    float2 _e60 = u_sm.m_0_;
    float2 _e62 = u_sm.m_1_;
    float2 l_u_c_v_1 = float2x2(_e60, _e62)[min(uint(_e57), 1u)];
    float l_u_e_cc_1 = u_sm.m_0_.x;
    int _e71 = idx_1;
    float l_u_e_cv_1 = u_sm.m_0_[_e71];
    int _e74 = idx_1;
    float2 _e77 = u_sm.m_0_;
    float2 _e79 = u_sm.m_1_;
    float l_u_e_vc_1 = float2x2(_e77, _e79)[min(uint(_e74), 1u)].x;
    int _e83 = idx_1;
    int _e84 = idx_1;
    float2 _e87 = u_sm.m_0_;
    float2 _e89 = u_sm.m_1_;
    float l_u_e_vv_1 = float2x2(_e87, _e89)[min(uint(_e83), 1u)][min(uint(_e84), 1u)];
    {
        StructWithMat _value2 = l_u_s;
        {
            float2x2 _value3 = _value2.m;
            s_sm.Store2(0+0, asuint(_value3[0]));
            s_sm.Store2(0+8, asuint(_value3[1]));
        }
    }
    {
        float2x2 _value2 = l_u_m_1;
        s_sm.Store2(0+0, asuint(_value2[0]));
        s_sm.Store2(0+8, asuint(_value2[1]));
    }
    s_sm.Store2(0+0, asuint(l_u_c_c_1));
    int _e100 = idx_1;
    s_sm.Store2(_e100*8+0, asuint(l_u_c_v_1));
    s_sm.Store(0+0+0, asuint(l_u_e_cc_1));
    int _e109 = idx_1;
    s_sm.Store(_e109*4+0+0, asuint(l_u_e_cv_1));
    int _e113 = idx_1;
    s_sm.Store(0+_e113*8+0, asuint(l_u_e_vc_1));
    int _e118 = idx_1;
    int _e120 = idx_1;
    s_sm.Store(_e120*4+_e118*8+0, asuint(l_u_e_vv_1));
    return;
}

typedef float2x2 ret_Constructarray4_float2x2_[4];
ret_Constructarray4_float2x2_ Constructarray4_float2x2_(float2x2 arg0, float2x2 arg1, float2x2 arg2, float2x2 arg3) {
    float2x2 ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef float2x2 ret_convert_std140_0_[4];
ret_convert_std140_0_ convert_std140_0_(mat2x2f32_std140_ v[4])
{
    return Constructarray4_float2x2_(float2x2(v[0].c_0_, v[0].c_1_), float2x2(v[1].c_0_, v[1].c_1_), float2x2(v[2].c_0_, v[2].c_1_), float2x2(v[3].c_0_, v[3].c_1_));
}

void access_am()
{
    int idx_2 = int(1);

    int _e2 = idx_2;
    idx_2 = asint(asuint(_e2) - asuint(int(1)));
    float2x2 l_s_a[4] = Constructarray4_float2x2_(float2x2(asfloat(s_am.Load2(0+0)), asfloat(s_am.Load2(0+8))), float2x2(asfloat(s_am.Load2(16+0)), asfloat(s_am.Load2(16+8))), float2x2(asfloat(s_am.Load2(32+0)), asfloat(s_am.Load2(32+8))), float2x2(asfloat(s_am.Load2(48+0)), asfloat(s_am.Load2(48+8))));
    float2x2 l_s_m_c = float2x2(asfloat(s_am.Load2(0+0)), asfloat(s_am.Load2(0+8)));
    int _e10 = idx_2;
    float2x2 l_s_m_v = float2x2(asfloat(s_am.Load2(_e10*16+0)), asfloat(s_am.Load2(_e10*16+8)));
    float2 l_s_c_cc = asfloat(s_am.Load2(0+0));
    int _e19 = idx_2;
    float2 l_s_c_cv = asfloat(s_am.Load2(_e19*8+0));
    int _e23 = idx_2;
    float2 l_s_c_vc = asfloat(s_am.Load2(0+_e23*16));
    int _e28 = idx_2;
    int _e30 = idx_2;
    float2 l_s_c_vv = asfloat(s_am.Load2(_e30*8+_e28*16));
    float l_s_e_ccc = asfloat(s_am.Load(0+0+0));
    int _e41 = idx_2;
    float l_s_e_ccv = asfloat(s_am.Load(_e41*4+0+0));
    int _e46 = idx_2;
    float l_s_e_cvc = asfloat(s_am.Load(0+_e46*8+0));
    int _e52 = idx_2;
    int _e54 = idx_2;
    float l_s_e_cvv = asfloat(s_am.Load(_e54*4+_e52*8+0));
    int _e58 = idx_2;
    float l_s_e_vcc = asfloat(s_am.Load(0+0+_e58*16));
    int _e64 = idx_2;
    int _e67 = idx_2;
    float l_s_e_vcv = asfloat(s_am.Load(_e67*4+0+_e64*16));
    int _e71 = idx_2;
    int _e73 = idx_2;
    float l_s_e_vvc = asfloat(s_am.Load(0+_e73*8+_e71*16));
    int _e78 = idx_2;
    int _e80 = idx_2;
    int _e82 = idx_2;
    float l_s_e_vvv = asfloat(s_am.Load(_e82*4+_e80*8+_e78*16));
    mat2x2f32_std140_ _e86[4] = u_am;
    const float2x2 _e264[4] = convert_std140_0_(_e86);
    float2 _e90 = u_am[0].c_0_;
    float2 _e92 = u_am[0].c_1_;
    float2x2 l_u_m_c = float2x2(_e90, _e92);
    int _e95 = idx_2;
    float2 _e98 = u_am[_e95].c_0_;
    float2 _e100 = u_am[_e95].c_1_;
    float2x2 l_u_m_v = float2x2(_e98, _e100);
    float2 l_u_c_cc = u_am[0].c_0_;
    int _e108 = idx_2;
    float2 _e110 = u_am[0].c_0_;
    float2 _e112 = u_am[0].c_1_;
    float2 l_u_c_cv = float2x2(_e110, _e112)[min(uint(_e108), 1u)];
    int _e116 = idx_2;
    float2 l_u_c_vc = u_am[_e116].c_0_;
    int _e121 = idx_2;
    int _e123 = idx_2;
    float2 _e125 = u_am[_e121].c_0_;
    float2 _e127 = u_am[_e121].c_1_;
    float2 l_u_c_vv = float2x2(_e125, _e127)[min(uint(_e123), 1u)];
    float l_u_e_ccc = u_am[0].c_0_.x;
    int _e138 = idx_2;
    float l_u_e_ccv = u_am[0].c_0_[_e138];
    int _e143 = idx_2;
    float2 _e145 = u_am[0].c_0_;
    float2 _e147 = u_am[0].c_1_;
    float l_u_e_cvc = float2x2(_e145, _e147)[min(uint(_e143), 1u)].x;
    int _e153 = idx_2;
    int _e154 = idx_2;
    float2 _e156 = u_am[0].c_0_;
    float2 _e158 = u_am[0].c_1_;
    float l_u_e_cvv = float2x2(_e156, _e158)[min(uint(_e153), 1u)][min(uint(_e154), 1u)];
    int _e163 = idx_2;
    float l_u_e_vcc = u_am[_e163].c_0_.x;
    int _e169 = idx_2;
    int _e172 = idx_2;
    float l_u_e_vcv = u_am[_e169].c_0_[_e172];
    int _e176 = idx_2;
    int _e178 = idx_2;
    float2 _e180 = u_am[_e176].c_0_;
    float2 _e182 = u_am[_e176].c_1_;
    float l_u_e_vvc = float2x2(_e180, _e182)[min(uint(_e178), 1u)].x;
    int _e187 = idx_2;
    int _e189 = idx_2;
    int _e190 = idx_2;
    float2 _e192 = u_am[_e187].c_0_;
    float2 _e194 = u_am[_e187].c_1_;
    float l_u_e_vvv = float2x2(_e192, _e194)[min(uint(_e189), 1u)][min(uint(_e190), 1u)];
    {
        float2x2 _value2[4] = _e264;
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
        float2x2 _value2 = l_u_m_c;
        s_am.Store2(0+0, asuint(_value2[0]));
        s_am.Store2(0+8, asuint(_value2[1]));
    }
    int _e201 = idx_2;
    {
        float2x2 _value2 = l_u_m_v;
        s_am.Store2(_e201*16+0, asuint(_value2[0]));
        s_am.Store2(_e201*16+8, asuint(_value2[1]));
    }
    s_am.Store2(0+0, asuint(l_u_c_cc));
    int _e208 = idx_2;
    s_am.Store2(_e208*8+0, asuint(l_u_c_cv));
    int _e211 = idx_2;
    s_am.Store2(0+_e211*16, asuint(l_u_c_vc));
    int _e215 = idx_2;
    int _e217 = idx_2;
    s_am.Store2(_e217*8+_e215*16, asuint(l_u_c_vv));
    s_am.Store(0+0+0, asuint(l_u_e_ccc));
    int _e226 = idx_2;
    s_am.Store(_e226*4+0+0, asuint(l_u_e_ccv));
    int _e230 = idx_2;
    s_am.Store(0+_e230*8+0, asuint(l_u_e_cvc));
    int _e235 = idx_2;
    int _e237 = idx_2;
    s_am.Store(_e237*4+_e235*8+0, asuint(l_u_e_cvv));
    int _e240 = idx_2;
    s_am.Store(0+0+_e240*16, asuint(l_u_e_vcc));
    int _e245 = idx_2;
    int _e248 = idx_2;
    s_am.Store(_e248*4+0+_e245*16, asuint(l_u_e_vcv));
    int _e251 = idx_2;
    int _e253 = idx_2;
    s_am.Store(0+_e253*8+_e251*16, asuint(l_u_e_vvc));
    int _e257 = idx_2;
    int _e259 = idx_2;
    int _e261 = idx_2;
    s_am.Store(_e261*4+_e259*8+_e257*16, asuint(l_u_e_vvv));
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
    return Constructarray4_StructWithMat_(ConstructStructWithMat(float2x2(v_1[0].m_0_, v_1[0].m_1_)), ConstructStructWithMat(float2x2(v_1[1].m_0_, v_1[1].m_1_)), ConstructStructWithMat(float2x2(v_1[2].m_0_, v_1[2].m_1_)), ConstructStructWithMat(float2x2(v_1[3].m_0_, v_1[3].m_1_)));
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
    StructWithArrayOfStructOfMat l_s_s_1 = ConstructStructWithArrayOfStructOfMat(Constructarray4_StructWithMat_(ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+0+0+0)), asfloat(s_sasm.Load2(0+0+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+16+0+0)), asfloat(s_sasm.Load2(0+16+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+32+0+0)), asfloat(s_sasm.Load2(0+32+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+48+0+0)), asfloat(s_sasm.Load2(0+48+0+8))))));
    StructWithMat l_s_a_1[4] = Constructarray4_StructWithMat_(ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+0+0+0)), asfloat(s_sasm.Load2(0+0+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+16+0+0)), asfloat(s_sasm.Load2(0+16+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+32+0+0)), asfloat(s_sasm.Load2(0+32+0+8)))), ConstructStructWithMat(float2x2(asfloat(s_sasm.Load2(0+48+0+0)), asfloat(s_sasm.Load2(0+48+0+8)))));
    float2x2 l_s_m_c_1 = float2x2(asfloat(s_sasm.Load2(0+0+0+0)), asfloat(s_sasm.Load2(0+0+0+8)));
    int _e16 = idx_3;
    float2x2 l_s_m_v_1 = float2x2(asfloat(s_sasm.Load2(0+_e16*16+0+0)), asfloat(s_sasm.Load2(0+_e16*16+0+8)));
    float2 l_s_c_cc_1 = asfloat(s_sasm.Load2(0+0+0+0));
    int _e30 = idx_3;
    float2 l_s_c_cv_1 = asfloat(s_sasm.Load2(_e30*8+0+0+0));
    int _e35 = idx_3;
    float2 l_s_c_vc_1 = asfloat(s_sasm.Load2(0+0+_e35*16+0));
    int _e42 = idx_3;
    int _e45 = idx_3;
    float2 l_s_c_vv_1 = asfloat(s_sasm.Load2(_e45*8+0+_e42*16+0));
    float l_s_e_ccc_1 = asfloat(s_sasm.Load(0+0+0+0+0));
    int _e60 = idx_3;
    float l_s_e_ccv_1 = asfloat(s_sasm.Load(_e60*4+0+0+0+0));
    int _e67 = idx_3;
    float l_s_e_cvc_1 = asfloat(s_sasm.Load(0+_e67*8+0+0+0));
    int _e75 = idx_3;
    int _e77 = idx_3;
    float l_s_e_cvv_1 = asfloat(s_sasm.Load(_e77*4+_e75*8+0+0+0));
    int _e82 = idx_3;
    float l_s_e_vcc_1 = asfloat(s_sasm.Load(0+0+0+_e82*16+0));
    int _e90 = idx_3;
    int _e94 = idx_3;
    float l_s_e_vcv_1 = asfloat(s_sasm.Load(_e94*4+0+0+_e90*16+0));
    int _e99 = idx_3;
    int _e102 = idx_3;
    float l_s_e_vvc_1 = asfloat(s_sasm.Load(0+_e102*8+0+_e99*16+0));
    int _e108 = idx_3;
    int _e111 = idx_3;
    int _e113 = idx_3;
    float l_s_e_vvv_1 = asfloat(s_sasm.Load(_e113*4+_e111*8+0+_e108*16+0));
    StructWithArrayOfStructOfMat_std140_ _e117 = u_sasm;
    const StructWithMat _e119[4] = convert_std140_1_(_e117.a);
    StructWithArrayOfStructOfMat l_u_s_1 = ConstructStructWithArrayOfStructOfMat(_e119);
    StructWithMat_std140_ _e123[4] = u_sasm.a;
    const StructWithMat _e345[4] = convert_std140_1_(_e123);
    float2 _e128 = u_sasm.a[0].m_0_;
    float2 _e130 = u_sasm.a[0].m_1_;
    float2x2 l_u_m_c_1 = float2x2(_e128, _e130);
    int _e134 = idx_3;
    float2 _e137 = u_sasm.a[_e134].m_0_;
    float2 _e139 = u_sasm.a[_e134].m_1_;
    float2x2 l_u_m_v_1 = float2x2(_e137, _e139);
    float2 l_u_c_cc_1 = u_sasm.a[0].m_0_;
    int _e149 = idx_3;
    float2 _e151 = u_sasm.a[0].m_0_;
    float2 _e153 = u_sasm.a[0].m_1_;
    float2 l_u_c_cv_1 = float2x2(_e151, _e153)[min(uint(_e149), 1u)];
    int _e158 = idx_3;
    float2 l_u_c_vc_1 = u_sasm.a[_e158].m_0_;
    int _e164 = idx_3;
    int _e166 = idx_3;
    float2 _e168 = u_sasm.a[_e164].m_0_;
    float2 _e170 = u_sasm.a[_e164].m_1_;
    float2 l_u_c_vv_1 = float2x2(_e168, _e170)[min(uint(_e166), 1u)];
    float l_u_e_ccc_1 = u_sasm.a[0].m_0_.x;
    int _e183 = idx_3;
    float l_u_e_ccv_1 = u_sasm.a[0].m_0_[_e183];
    int _e189 = idx_3;
    float2 _e191 = u_sasm.a[0].m_0_;
    float2 _e193 = u_sasm.a[0].m_1_;
    float l_u_e_cvc_1 = float2x2(_e191, _e193)[min(uint(_e189), 1u)].x;
    int _e200 = idx_3;
    int _e201 = idx_3;
    float2 _e203 = u_sasm.a[0].m_0_;
    float2 _e205 = u_sasm.a[0].m_1_;
    float l_u_e_cvv_1 = float2x2(_e203, _e205)[min(uint(_e200), 1u)][min(uint(_e201), 1u)];
    int _e211 = idx_3;
    float l_u_e_vcc_1 = u_sasm.a[_e211].m_0_.x;
    int _e218 = idx_3;
    int _e221 = idx_3;
    float l_u_e_vcv_1 = u_sasm.a[_e218].m_0_[_e221];
    int _e226 = idx_3;
    int _e228 = idx_3;
    float2 _e230 = u_sasm.a[_e226].m_0_;
    float2 _e232 = u_sasm.a[_e226].m_1_;
    float l_u_e_vvc_1 = float2x2(_e230, _e232)[min(uint(_e228), 1u)].x;
    int _e238 = idx_3;
    int _e240 = idx_3;
    int _e241 = idx_3;
    float2 _e243 = u_sasm.a[_e238].m_0_;
    float2 _e245 = u_sasm.a[_e238].m_1_;
    float l_u_e_vvv_1 = float2x2(_e243, _e245)[min(uint(_e240), 1u)][min(uint(_e241), 1u)];
    {
        StructWithArrayOfStructOfMat _value2 = l_u_s_1;
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
        StructWithMat _value2[4] = _e345;
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
        float2x2 _value2 = l_u_m_c_1;
        s_sasm.Store2(0+0+0+0, asuint(_value2[0]));
        s_sasm.Store2(0+0+0+8, asuint(_value2[1]));
    }
    int _e257 = idx_3;
    {
        float2x2 _value2 = l_u_m_v_1;
        s_sasm.Store2(0+_e257*16+0+0, asuint(_value2[0]));
        s_sasm.Store2(0+_e257*16+0+8, asuint(_value2[1]));
    }
    s_sasm.Store2(0+0+0+0, asuint(l_u_c_cc_1));
    int _e269 = idx_3;
    s_sasm.Store2(_e269*8+0+0+0, asuint(l_u_c_cv_1));
    int _e273 = idx_3;
    s_sasm.Store2(0+0+_e273*16+0, asuint(l_u_c_vc_1));
    int _e279 = idx_3;
    int _e282 = idx_3;
    s_sasm.Store2(_e282*8+0+_e279*16+0, asuint(l_u_c_vv_1));
    s_sasm.Store(0+0+0+0+0, asuint(l_u_e_ccc_1));
    int _e295 = idx_3;
    s_sasm.Store(_e295*4+0+0+0+0, asuint(l_u_e_ccv_1));
    int _e301 = idx_3;
    s_sasm.Store(0+_e301*8+0+0+0, asuint(l_u_e_cvc_1));
    int _e308 = idx_3;
    int _e310 = idx_3;
    s_sasm.Store(_e310*4+_e308*8+0+0+0, asuint(l_u_e_cvv_1));
    int _e314 = idx_3;
    s_sasm.Store(0+0+0+_e314*16+0, asuint(l_u_e_vcc_1));
    int _e321 = idx_3;
    int _e325 = idx_3;
    s_sasm.Store(_e325*4+0+0+_e321*16+0, asuint(l_u_e_vcv_1));
    int _e329 = idx_3;
    int _e332 = idx_3;
    s_sasm.Store(0+_e332*8+0+_e329*16+0, asuint(l_u_e_vvc_1));
    int _e337 = idx_3;
    int _e340 = idx_3;
    int _e342 = idx_3;
    s_sasm.Store(_e342*4+_e340*8+0+_e337*16+0, asuint(l_u_e_vvv_1));
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
