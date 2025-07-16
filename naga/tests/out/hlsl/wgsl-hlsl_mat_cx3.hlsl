struct StructWithMat {
    row_major float3x3 m;
    int _end_pad_0;
};

struct StructWithArrayOfStructOfMat {
    StructWithMat a[4];
};

RWByteAddressBuffer s_m : register(u0);
cbuffer u_m : register(b1) { row_major float3x3 u_m; }
RWByteAddressBuffer s_sm : register(u0, space1);
cbuffer u_sm : register(b1, space1) { StructWithMat u_sm; }
RWByteAddressBuffer s_sasm : register(u0, space2);
cbuffer u_sasm : register(b1, space2) { StructWithArrayOfStructOfMat u_sasm; }

void access_m()
{
    int idx = int(1);

    int _e3 = idx;
    idx = asint(asuint(_e3) - asuint(int(1)));
    float3x3 l_s_m = float3x3(asfloat(s_m.Load3(0)), asfloat(s_m.Load3(16)), asfloat(s_m.Load3(32)));
    float3 l_s_c_c = asfloat(s_m.Load3(0));
    int _e11 = idx;
    float3 l_s_c_v = asfloat(s_m.Load3(_e11*16));
    float l_s_e_cc = asfloat(s_m.Load(0+0));
    int _e20 = idx;
    float l_s_e_cv = asfloat(s_m.Load(_e20*4+0));
    int _e24 = idx;
    float l_s_e_vc = asfloat(s_m.Load(0+_e24*16));
    int _e29 = idx;
    int _e31 = idx;
    float l_s_e_vv = asfloat(s_m.Load(_e31*4+_e29*16));
    float3x3 l_u_m = u_m;
    float3 l_u_c_c = u_m[0];
    int _e40 = idx;
    float3 l_u_c_v = u_m[_e40];
    float l_u_e_cc = u_m[0].x;
    int _e49 = idx;
    float l_u_e_cv = u_m[0][_e49];
    int _e53 = idx;
    float l_u_e_vc = u_m[_e53].x;
    int _e58 = idx;
    int _e60 = idx;
    float l_u_e_vv = u_m[_e58][_e60];
    {
        float3x3 _value2 = l_u_m;
        s_m.Store3(0, asuint(_value2[0]));
        s_m.Store3(16, asuint(_value2[1]));
        s_m.Store3(32, asuint(_value2[2]));
    }
    s_m.Store3(0, asuint(l_u_c_c));
    int _e67 = idx;
    s_m.Store3(_e67*16, asuint(l_u_c_v));
    s_m.Store(0+0, asuint(l_u_e_cc));
    int _e74 = idx;
    s_m.Store(_e74*4+0, asuint(l_u_e_cv));
    int _e77 = idx;
    s_m.Store(0+_e77*16, asuint(l_u_e_vc));
    int _e81 = idx;
    int _e83 = idx;
    s_m.Store(_e83*4+_e81*16, asuint(l_u_e_vv));
    return;
}

StructWithMat ConstructStructWithMat(float3x3 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m = arg0;
    return ret;
}

void access_sm()
{
    int idx_1 = int(1);

    int _e3 = idx_1;
    idx_1 = asint(asuint(_e3) - asuint(int(1)));
    StructWithMat l_s_s = ConstructStructWithMat(float3x3(asfloat(s_sm.Load3(0+0)), asfloat(s_sm.Load3(0+16)), asfloat(s_sm.Load3(0+32))));
    float3x3 l_s_m_1 = float3x3(asfloat(s_sm.Load3(0+0)), asfloat(s_sm.Load3(0+16)), asfloat(s_sm.Load3(0+32)));
    float3 l_s_c_c_1 = asfloat(s_sm.Load3(0+0));
    int _e16 = idx_1;
    float3 l_s_c_v_1 = asfloat(s_sm.Load3(_e16*16+0));
    float l_s_e_cc_1 = asfloat(s_sm.Load(0+0+0));
    int _e27 = idx_1;
    float l_s_e_cv_1 = asfloat(s_sm.Load(_e27*4+0+0));
    int _e32 = idx_1;
    float l_s_e_vc_1 = asfloat(s_sm.Load(0+_e32*16+0));
    int _e38 = idx_1;
    int _e40 = idx_1;
    float l_s_e_vv_1 = asfloat(s_sm.Load(_e40*4+_e38*16+0));
    StructWithMat l_u_s = u_sm;
    float3x3 l_u_m_1 = u_sm.m;
    float3 l_u_c_c_1 = u_sm.m[0];
    int _e54 = idx_1;
    float3 l_u_c_v_1 = u_sm.m[_e54];
    float l_u_e_cc_1 = u_sm.m[0].x;
    int _e65 = idx_1;
    float l_u_e_cv_1 = u_sm.m[0][_e65];
    int _e70 = idx_1;
    float l_u_e_vc_1 = u_sm.m[_e70].x;
    int _e76 = idx_1;
    int _e78 = idx_1;
    float l_u_e_vv_1 = u_sm.m[_e76][_e78];
    {
        StructWithMat _value2 = l_u_s;
        {
            float3x3 _value3 = _value2.m;
            s_sm.Store3(0+0, asuint(_value3[0]));
            s_sm.Store3(0+16, asuint(_value3[1]));
            s_sm.Store3(0+32, asuint(_value3[2]));
        }
    }
    {
        float3x3 _value2 = l_u_m_1;
        s_sm.Store3(0+0, asuint(_value2[0]));
        s_sm.Store3(0+16, asuint(_value2[1]));
        s_sm.Store3(0+32, asuint(_value2[2]));
    }
    s_sm.Store3(0+0, asuint(l_u_c_c_1));
    int _e89 = idx_1;
    s_sm.Store3(_e89*16+0, asuint(l_u_c_v_1));
    s_sm.Store(0+0+0, asuint(l_u_e_cc_1));
    int _e98 = idx_1;
    s_sm.Store(_e98*4+0+0, asuint(l_u_e_cv_1));
    int _e102 = idx_1;
    s_sm.Store(0+_e102*16+0, asuint(l_u_e_vc_1));
    int _e107 = idx_1;
    int _e109 = idx_1;
    s_sm.Store(_e109*4+_e107*16+0, asuint(l_u_e_vv_1));
    return;
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

void access_sasm()
{
    int idx_2 = int(1);

    int _e3 = idx_2;
    idx_2 = asint(asuint(_e3) - asuint(int(1)));
    StructWithArrayOfStructOfMat l_s_s_1 = ConstructStructWithArrayOfStructOfMat(Constructarray4_StructWithMat_(ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+0+0+0)), asfloat(s_sasm.Load3(0+0+0+16)), asfloat(s_sasm.Load3(0+0+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+48+0+0)), asfloat(s_sasm.Load3(0+48+0+16)), asfloat(s_sasm.Load3(0+48+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+96+0+0)), asfloat(s_sasm.Load3(0+96+0+16)), asfloat(s_sasm.Load3(0+96+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+144+0+0)), asfloat(s_sasm.Load3(0+144+0+16)), asfloat(s_sasm.Load3(0+144+0+32))))));
    StructWithMat l_s_a[4] = Constructarray4_StructWithMat_(ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+0+0+0)), asfloat(s_sasm.Load3(0+0+0+16)), asfloat(s_sasm.Load3(0+0+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+48+0+0)), asfloat(s_sasm.Load3(0+48+0+16)), asfloat(s_sasm.Load3(0+48+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+96+0+0)), asfloat(s_sasm.Load3(0+96+0+16)), asfloat(s_sasm.Load3(0+96+0+32)))), ConstructStructWithMat(float3x3(asfloat(s_sasm.Load3(0+144+0+0)), asfloat(s_sasm.Load3(0+144+0+16)), asfloat(s_sasm.Load3(0+144+0+32)))));
    float3x3 l_s_m_c = float3x3(asfloat(s_sasm.Load3(0+0+0+0)), asfloat(s_sasm.Load3(0+0+0+16)), asfloat(s_sasm.Load3(0+0+0+32)));
    int _e17 = idx_2;
    float3x3 l_s_m_v = float3x3(asfloat(s_sasm.Load3(0+_e17*48+0+0)), asfloat(s_sasm.Load3(0+_e17*48+0+16)), asfloat(s_sasm.Load3(0+_e17*48+0+32)));
    float3 l_s_c_cc = asfloat(s_sasm.Load3(0+0+0+0));
    int _e31 = idx_2;
    float3 l_s_c_cv = asfloat(s_sasm.Load3(_e31*16+0+0+0));
    int _e36 = idx_2;
    float3 l_s_c_vc = asfloat(s_sasm.Load3(0+0+_e36*48+0));
    int _e43 = idx_2;
    int _e46 = idx_2;
    float3 l_s_c_vv = asfloat(s_sasm.Load3(_e46*16+0+_e43*48+0));
    float l_s_e_ccc = asfloat(s_sasm.Load(0+0+0+0+0));
    int _e61 = idx_2;
    float l_s_e_ccv = asfloat(s_sasm.Load(_e61*4+0+0+0+0));
    int _e68 = idx_2;
    float l_s_e_cvc = asfloat(s_sasm.Load(0+_e68*16+0+0+0));
    int _e76 = idx_2;
    int _e78 = idx_2;
    float l_s_e_cvv = asfloat(s_sasm.Load(_e78*4+_e76*16+0+0+0));
    int _e83 = idx_2;
    float l_s_e_vcc = asfloat(s_sasm.Load(0+0+0+_e83*48+0));
    int _e91 = idx_2;
    int _e95 = idx_2;
    float l_s_e_vcv = asfloat(s_sasm.Load(_e95*4+0+0+_e91*48+0));
    int _e100 = idx_2;
    int _e103 = idx_2;
    float l_s_e_vvc = asfloat(s_sasm.Load(0+_e103*16+0+_e100*48+0));
    int _e109 = idx_2;
    int _e112 = idx_2;
    int _e114 = idx_2;
    float l_s_e_vvv = asfloat(s_sasm.Load(_e114*4+_e112*16+0+_e109*48+0));
    StructWithArrayOfStructOfMat l_u_s_1 = u_sasm;
    StructWithMat l_u_a[4] = u_sasm.a;
    float3x3 l_u_m_c = u_sasm.a[0].m;
    int _e129 = idx_2;
    float3x3 l_u_m_v = u_sasm.a[_e129].m;
    float3 l_u_c_cc = u_sasm.a[0].m[0];
    int _e143 = idx_2;
    float3 l_u_c_cv = u_sasm.a[0].m[_e143];
    int _e148 = idx_2;
    float3 l_u_c_vc = u_sasm.a[_e148].m[0];
    int _e155 = idx_2;
    int _e158 = idx_2;
    float3 l_u_c_vv = u_sasm.a[_e155].m[_e158];
    float l_u_e_ccc = u_sasm.a[0].m[0].x;
    int _e173 = idx_2;
    float l_u_e_ccv = u_sasm.a[0].m[0][_e173];
    int _e180 = idx_2;
    float l_u_e_cvc = u_sasm.a[0].m[_e180].x;
    int _e188 = idx_2;
    int _e190 = idx_2;
    float l_u_e_cvv = u_sasm.a[0].m[_e188][_e190];
    int _e195 = idx_2;
    float l_u_e_vcc = u_sasm.a[_e195].m[0].x;
    int _e203 = idx_2;
    int _e207 = idx_2;
    float l_u_e_vcv = u_sasm.a[_e203].m[0][_e207];
    int _e212 = idx_2;
    int _e215 = idx_2;
    float l_u_e_vvc = u_sasm.a[_e212].m[_e215].x;
    int _e221 = idx_2;
    int _e224 = idx_2;
    int _e226 = idx_2;
    float l_u_e_vvv = u_sasm.a[_e221].m[_e224][_e226];
    {
        StructWithArrayOfStructOfMat _value2 = l_u_s_1;
        {
            StructWithMat _value3[4] = _value2.a;
            {
                StructWithMat _value4 = _value3[0];
                {
                    float3x3 _value5 = _value4.m;
                    s_sasm.Store3(0+0+0+0, asuint(_value5[0]));
                    s_sasm.Store3(0+0+0+16, asuint(_value5[1]));
                    s_sasm.Store3(0+0+0+32, asuint(_value5[2]));
                }
            }
            {
                StructWithMat _value4 = _value3[1];
                {
                    float3x3 _value5 = _value4.m;
                    s_sasm.Store3(0+48+0+0, asuint(_value5[0]));
                    s_sasm.Store3(0+48+0+16, asuint(_value5[1]));
                    s_sasm.Store3(0+48+0+32, asuint(_value5[2]));
                }
            }
            {
                StructWithMat _value4 = _value3[2];
                {
                    float3x3 _value5 = _value4.m;
                    s_sasm.Store3(0+96+0+0, asuint(_value5[0]));
                    s_sasm.Store3(0+96+0+16, asuint(_value5[1]));
                    s_sasm.Store3(0+96+0+32, asuint(_value5[2]));
                }
            }
            {
                StructWithMat _value4 = _value3[3];
                {
                    float3x3 _value5 = _value4.m;
                    s_sasm.Store3(0+144+0+0, asuint(_value5[0]));
                    s_sasm.Store3(0+144+0+16, asuint(_value5[1]));
                    s_sasm.Store3(0+144+0+32, asuint(_value5[2]));
                }
            }
        }
    }
    {
        StructWithMat _value2[4] = l_u_a;
        {
            StructWithMat _value3 = _value2[0];
            {
                float3x3 _value4 = _value3.m;
                s_sasm.Store3(0+0+0+0, asuint(_value4[0]));
                s_sasm.Store3(0+0+0+16, asuint(_value4[1]));
                s_sasm.Store3(0+0+0+32, asuint(_value4[2]));
            }
        }
        {
            StructWithMat _value3 = _value2[1];
            {
                float3x3 _value4 = _value3.m;
                s_sasm.Store3(0+48+0+0, asuint(_value4[0]));
                s_sasm.Store3(0+48+0+16, asuint(_value4[1]));
                s_sasm.Store3(0+48+0+32, asuint(_value4[2]));
            }
        }
        {
            StructWithMat _value3 = _value2[2];
            {
                float3x3 _value4 = _value3.m;
                s_sasm.Store3(0+96+0+0, asuint(_value4[0]));
                s_sasm.Store3(0+96+0+16, asuint(_value4[1]));
                s_sasm.Store3(0+96+0+32, asuint(_value4[2]));
            }
        }
        {
            StructWithMat _value3 = _value2[3];
            {
                float3x3 _value4 = _value3.m;
                s_sasm.Store3(0+144+0+0, asuint(_value4[0]));
                s_sasm.Store3(0+144+0+16, asuint(_value4[1]));
                s_sasm.Store3(0+144+0+32, asuint(_value4[2]));
            }
        }
    }
    {
        float3x3 _value2 = l_u_m_c;
        s_sasm.Store3(0+0+0+0, asuint(_value2[0]));
        s_sasm.Store3(0+0+0+16, asuint(_value2[1]));
        s_sasm.Store3(0+0+0+32, asuint(_value2[2]));
    }
    int _e238 = idx_2;
    {
        float3x3 _value2 = l_u_m_v;
        s_sasm.Store3(0+_e238*48+0+0, asuint(_value2[0]));
        s_sasm.Store3(0+_e238*48+0+16, asuint(_value2[1]));
        s_sasm.Store3(0+_e238*48+0+32, asuint(_value2[2]));
    }
    s_sasm.Store3(0+0+0+0, asuint(l_u_c_cc));
    int _e250 = idx_2;
    s_sasm.Store3(_e250*16+0+0+0, asuint(l_u_c_cv));
    int _e254 = idx_2;
    s_sasm.Store3(0+0+_e254*48+0, asuint(l_u_c_vc));
    int _e260 = idx_2;
    int _e263 = idx_2;
    s_sasm.Store3(_e263*16+0+_e260*48+0, asuint(l_u_c_vv));
    s_sasm.Store(0+0+0+0+0, asuint(l_u_e_ccc));
    int _e276 = idx_2;
    s_sasm.Store(_e276*4+0+0+0+0, asuint(l_u_e_ccv));
    int _e282 = idx_2;
    s_sasm.Store(0+_e282*16+0+0+0, asuint(l_u_e_cvc));
    int _e289 = idx_2;
    int _e291 = idx_2;
    s_sasm.Store(_e291*4+_e289*16+0+0+0, asuint(l_u_e_cvv));
    int _e295 = idx_2;
    s_sasm.Store(0+0+0+_e295*48+0, asuint(l_u_e_vcc));
    int _e302 = idx_2;
    int _e306 = idx_2;
    s_sasm.Store(_e306*4+0+0+_e302*48+0, asuint(l_u_e_vcv));
    int _e310 = idx_2;
    int _e313 = idx_2;
    s_sasm.Store(0+_e313*16+0+_e310*48+0, asuint(l_u_e_vvc));
    int _e318 = idx_2;
    int _e321 = idx_2;
    int _e323 = idx_2;
    s_sasm.Store(_e323*4+_e321*16+0+_e318*48+0, asuint(l_u_e_vvv));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    access_m();
    access_sm();
    access_sasm();
    return;
}
