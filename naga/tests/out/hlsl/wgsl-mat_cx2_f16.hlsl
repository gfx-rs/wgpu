typedef struct { half2 _0; half2 _1; half2 _2; half2 _3; } __mat4x2_f16;
half2 __get_col_of_mat4x2_f16(__mat4x2_f16 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    case 3: { return mat._3; }
    default: { return (half2)0; }
    }
}
void __set_col_of_mat4x2_f16(__mat4x2_f16 mat, uint idx, half2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    case 2: { mat._2 = value; break; }
    case 3: { mat._3 = value; break; }
    }
}
void __set_el_of_mat4x2_f16(__mat4x2_f16 mat, uint idx, uint vec_idx, half value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    case 2: { mat._2[vec_idx] = value; break; }
    case 3: { mat._3[vec_idx] = value; break; }
    }
}

struct StructWithMat {
    half2 m_0; half2 m_1; half2 m_2; half2 m_3;
};

struct StructWithArrayOfStructOfMat {
    StructWithMat a[4];
};

RWByteAddressBuffer s_m : register(u0);
cbuffer u_m : register(b1) { __mat4x2_f16 u_m; }
RWByteAddressBuffer s_sm : register(u0, space1);
cbuffer u_sm : register(b1, space1) { StructWithMat u_sm; }
RWByteAddressBuffer s_am : register(u0, space2);
cbuffer u_am : register(b1, space2) { __mat4x2_f16 u_am[4]; }
RWByteAddressBuffer s_sasm : register(u0, space3);
cbuffer u_sasm : register(b1, space3) { StructWithArrayOfStructOfMat u_sasm; }

void access_m()
{
    int idx = int(1);

    int _e3 = idx;
    idx = asint(asuint(_e3) - asuint(int(1)));
    half4x2 l_s_m = half4x2(s_m.Load<half2>(0), s_m.Load<half2>(4), s_m.Load<half2>(8), s_m.Load<half2>(12));
    half2 l_s_c_c = s_m.Load<half2>(0);
    int _e11 = idx;
    half2 l_s_c_v = s_m.Load<half2>(_e11*4);
    half l_s_e_cc = s_m.Load<half>(0+0);
    int _e20 = idx;
    half l_s_e_cv = s_m.Load<half>(_e20*2+0);
    int _e24 = idx;
    half l_s_e_vc = s_m.Load<half>(0+_e24*4);
    int _e29 = idx;
    int _e31 = idx;
    half l_s_e_vv = s_m.Load<half>(_e31*2+_e29*4);
    half4x2 l_u_m = ((half4x2)u_m);
    half2 l_u_c_c = u_m._0;
    int _e40 = idx;
    half2 l_u_c_v = __get_col_of_mat4x2_f16(u_m, _e40);
    half l_u_e_cc = u_m._0.x;
    int _e49 = idx;
    half l_u_e_cv = u_m._0[_e49];
    int _e53 = idx;
    half l_u_e_vc = __get_col_of_mat4x2_f16(u_m, _e53).x;
    int _e58 = idx;
    int _e60 = idx;
    half l_u_e_vv = __get_col_of_mat4x2_f16(u_m, _e58)[_e60];
    {
        half4x2 _value2 = l_u_m;
        s_m.Store(0, _value2[0]);
        s_m.Store(4, _value2[1]);
        s_m.Store(8, _value2[2]);
        s_m.Store(12, _value2[3]);
    }
    s_m.Store(0, l_u_c_c);
    int _e67 = idx;
    s_m.Store(_e67*4, l_u_c_v);
    s_m.Store(0+0, l_u_e_cc);
    int _e74 = idx;
    s_m.Store(_e74*2+0, l_u_e_cv);
    int _e77 = idx;
    s_m.Store(0+_e77*4, l_u_e_vc);
    int _e81 = idx;
    int _e83 = idx;
    s_m.Store(_e83*2+_e81*4, l_u_e_vv);
    return;
}

StructWithMat ConstructStructWithMat(half4x2 arg0) {
    StructWithMat ret = (StructWithMat)0;
    ret.m_0 = arg0[0];
    ret.m_1 = arg0[1];
    ret.m_2 = arg0[2];
    ret.m_3 = arg0[3];
    return ret;
}

half4x2 GetMatmOnStructWithMat(StructWithMat obj) {
    return half4x2(obj.m_0, obj.m_1, obj.m_2, obj.m_3);
}

void SetMatmOnStructWithMat(StructWithMat obj, half4x2 mat) {
    obj.m_0 = mat[0];
    obj.m_1 = mat[1];
    obj.m_2 = mat[2];
    obj.m_3 = mat[3];
}

void SetMatVecmOnStructWithMat(StructWithMat obj, half2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0 = vec; break; }
    case 1: { obj.m_1 = vec; break; }
    case 2: { obj.m_2 = vec; break; }
    case 3: { obj.m_3 = vec; break; }
    }
}

void SetMatScalarmOnStructWithMat(StructWithMat obj, half scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0[vec_idx] = scalar; break; }
    case 1: { obj.m_1[vec_idx] = scalar; break; }
    case 2: { obj.m_2[vec_idx] = scalar; break; }
    case 3: { obj.m_3[vec_idx] = scalar; break; }
    }
}

void access_sm()
{
    int idx_1 = int(1);

    int _e3 = idx_1;
    idx_1 = asint(asuint(_e3) - asuint(int(1)));
    StructWithMat l_s_s = ConstructStructWithMat(half4x2(s_sm.Load<half2>(0+0), s_sm.Load<half2>(0+4), s_sm.Load<half2>(0+8), s_sm.Load<half2>(0+12)));
    half4x2 l_s_m_1 = half4x2(s_sm.Load<half2>(0+0), s_sm.Load<half2>(0+4), s_sm.Load<half2>(0+8), s_sm.Load<half2>(0+12));
    half2 l_s_c_c_1 = s_sm.Load<half2>(0+0);
    int _e16 = idx_1;
    half2 l_s_c_v_1 = s_sm.Load<half2>(_e16*4+0);
    half l_s_e_cc_1 = s_sm.Load<half>(0+0+0);
    int _e27 = idx_1;
    half l_s_e_cv_1 = s_sm.Load<half>(_e27*2+0+0);
    int _e32 = idx_1;
    half l_s_e_vc_1 = s_sm.Load<half>(0+_e32*4+0);
    int _e38 = idx_1;
    int _e40 = idx_1;
    half l_s_e_vv_1 = s_sm.Load<half>(_e40*2+_e38*4+0);
    StructWithMat l_u_s = u_sm;
    half4x2 l_u_m_1 = GetMatmOnStructWithMat(u_sm);
    half2 l_u_c_c_1 = GetMatmOnStructWithMat(u_sm)[0];
    int _e54 = idx_1;
    half2 l_u_c_v_1 = GetMatmOnStructWithMat(u_sm)[_e54];
    half l_u_e_cc_1 = GetMatmOnStructWithMat(u_sm)[0].x;
    int _e65 = idx_1;
    half l_u_e_cv_1 = GetMatmOnStructWithMat(u_sm)[0][_e65];
    int _e70 = idx_1;
    half l_u_e_vc_1 = GetMatmOnStructWithMat(u_sm)[_e70].x;
    int _e76 = idx_1;
    int _e78 = idx_1;
    half l_u_e_vv_1 = GetMatmOnStructWithMat(u_sm)[_e76][_e78];
    {
        StructWithMat _value2 = l_u_s;
        {
            s_sm.Store(0+0, _value2.m_0);
            s_sm.Store(0+4, _value2.m_1);
            s_sm.Store(0+8, _value2.m_2);
            s_sm.Store(0+12, _value2.m_3);
        }
    }
    {
        half4x2 _value2 = l_u_m_1;
        s_sm.Store(0+0, _value2[0]);
        s_sm.Store(0+4, _value2[1]);
        s_sm.Store(0+8, _value2[2]);
        s_sm.Store(0+12, _value2[3]);
    }
    s_sm.Store(0+0, l_u_c_c_1);
    int _e89 = idx_1;
    s_sm.Store(_e89*4+0, l_u_c_v_1);
    s_sm.Store(0+0+0, l_u_e_cc_1);
    int _e98 = idx_1;
    s_sm.Store(_e98*2+0+0, l_u_e_cv_1);
    int _e102 = idx_1;
    s_sm.Store(0+_e102*4+0, l_u_e_vc_1);
    int _e107 = idx_1;
    int _e109 = idx_1;
    s_sm.Store(_e109*2+_e107*4+0, l_u_e_vv_1);
    return;
}

typedef half4x2 ret_Constructarray4_half4x2_[4];
ret_Constructarray4_half4x2_ Constructarray4_half4x2_(half4x2 arg0, half4x2 arg1, half4x2 arg2, half4x2 arg3) {
    half4x2 ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

void access_am()
{
    int idx_2 = int(1);

    int _e3 = idx_2;
    idx_2 = asint(asuint(_e3) - asuint(int(1)));
    half4x2 l_s_a[4] = Constructarray4_half4x2_(half4x2(s_am.Load<half2>(0+0), s_am.Load<half2>(0+4), s_am.Load<half2>(0+8), s_am.Load<half2>(0+12)), half4x2(s_am.Load<half2>(16+0), s_am.Load<half2>(16+4), s_am.Load<half2>(16+8), s_am.Load<half2>(16+12)), half4x2(s_am.Load<half2>(32+0), s_am.Load<half2>(32+4), s_am.Load<half2>(32+8), s_am.Load<half2>(32+12)), half4x2(s_am.Load<half2>(48+0), s_am.Load<half2>(48+4), s_am.Load<half2>(48+8), s_am.Load<half2>(48+12)));
    half4x2 l_s_m_c = half4x2(s_am.Load<half2>(0+0), s_am.Load<half2>(0+4), s_am.Load<half2>(0+8), s_am.Load<half2>(0+12));
    int _e11 = idx_2;
    half4x2 l_s_m_v = half4x2(s_am.Load<half2>(_e11*16+0), s_am.Load<half2>(_e11*16+4), s_am.Load<half2>(_e11*16+8), s_am.Load<half2>(_e11*16+12));
    half2 l_s_c_cc = s_am.Load<half2>(0+0);
    int _e20 = idx_2;
    half2 l_s_c_cv = s_am.Load<half2>(_e20*4+0);
    int _e24 = idx_2;
    half2 l_s_c_vc = s_am.Load<half2>(0+_e24*16);
    int _e29 = idx_2;
    int _e31 = idx_2;
    half2 l_s_c_vv = s_am.Load<half2>(_e31*4+_e29*16);
    half l_s_e_ccc = s_am.Load<half>(0+0+0);
    int _e42 = idx_2;
    half l_s_e_ccv = s_am.Load<half>(_e42*2+0+0);
    int _e47 = idx_2;
    half l_s_e_cvc = s_am.Load<half>(0+_e47*4+0);
    int _e53 = idx_2;
    int _e55 = idx_2;
    half l_s_e_cvv = s_am.Load<half>(_e55*2+_e53*4+0);
    int _e59 = idx_2;
    half l_s_e_vcc = s_am.Load<half>(0+0+_e59*16);
    int _e65 = idx_2;
    int _e68 = idx_2;
    half l_s_e_vcv = s_am.Load<half>(_e68*2+0+_e65*16);
    int _e72 = idx_2;
    int _e74 = idx_2;
    half l_s_e_vvc = s_am.Load<half>(0+_e74*4+_e72*16);
    int _e79 = idx_2;
    int _e81 = idx_2;
    int _e83 = idx_2;
    half l_s_e_vvv = s_am.Load<half>(_e83*2+_e81*4+_e79*16);
    half4x2 l_u_a[4] = ((half4x2[4])u_am);
    half4x2 l_u_m_c = ((half4x2)u_am[0]);
    int _e92 = idx_2;
    half4x2 l_u_m_v = ((half4x2)u_am[_e92]);
    half2 l_u_c_cc = u_am[0]._0;
    int _e101 = idx_2;
    half2 l_u_c_cv = __get_col_of_mat4x2_f16(u_am[0], _e101);
    int _e105 = idx_2;
    half2 l_u_c_vc = u_am[_e105]._0;
    int _e110 = idx_2;
    int _e112 = idx_2;
    half2 l_u_c_vv = __get_col_of_mat4x2_f16(u_am[_e110], _e112);
    half l_u_e_ccc = u_am[0]._0.x;
    int _e123 = idx_2;
    half l_u_e_ccv = u_am[0]._0[_e123];
    int _e128 = idx_2;
    half l_u_e_cvc = __get_col_of_mat4x2_f16(u_am[0], _e128).x;
    int _e134 = idx_2;
    int _e136 = idx_2;
    half l_u_e_cvv = __get_col_of_mat4x2_f16(u_am[0], _e134)[_e136];
    int _e140 = idx_2;
    half l_u_e_vcc = u_am[_e140]._0.x;
    int _e146 = idx_2;
    int _e149 = idx_2;
    half l_u_e_vcv = u_am[_e146]._0[_e149];
    int _e153 = idx_2;
    int _e155 = idx_2;
    half l_u_e_vvc = __get_col_of_mat4x2_f16(u_am[_e153], _e155).x;
    int _e160 = idx_2;
    int _e162 = idx_2;
    int _e164 = idx_2;
    half l_u_e_vvv = __get_col_of_mat4x2_f16(u_am[_e160], _e162)[_e164];
    {
        half4x2 _value2[4] = l_u_a;
        {
            half4x2 _value3 = _value2[0];
            s_am.Store(0+0, _value3[0]);
            s_am.Store(0+4, _value3[1]);
            s_am.Store(0+8, _value3[2]);
            s_am.Store(0+12, _value3[3]);
        }
        {
            half4x2 _value3 = _value2[1];
            s_am.Store(16+0, _value3[0]);
            s_am.Store(16+4, _value3[1]);
            s_am.Store(16+8, _value3[2]);
            s_am.Store(16+12, _value3[3]);
        }
        {
            half4x2 _value3 = _value2[2];
            s_am.Store(32+0, _value3[0]);
            s_am.Store(32+4, _value3[1]);
            s_am.Store(32+8, _value3[2]);
            s_am.Store(32+12, _value3[3]);
        }
        {
            half4x2 _value3 = _value2[3];
            s_am.Store(48+0, _value3[0]);
            s_am.Store(48+4, _value3[1]);
            s_am.Store(48+8, _value3[2]);
            s_am.Store(48+12, _value3[3]);
        }
    }
    {
        half4x2 _value2 = l_u_m_c;
        s_am.Store(0+0, _value2[0]);
        s_am.Store(0+4, _value2[1]);
        s_am.Store(0+8, _value2[2]);
        s_am.Store(0+12, _value2[3]);
    }
    int _e171 = idx_2;
    {
        half4x2 _value2 = l_u_m_v;
        s_am.Store(_e171*16+0, _value2[0]);
        s_am.Store(_e171*16+4, _value2[1]);
        s_am.Store(_e171*16+8, _value2[2]);
        s_am.Store(_e171*16+12, _value2[3]);
    }
    s_am.Store(0+0, l_u_c_cc);
    int _e178 = idx_2;
    s_am.Store(_e178*4+0, l_u_c_cv);
    int _e181 = idx_2;
    s_am.Store(0+_e181*16, l_u_c_vc);
    int _e185 = idx_2;
    int _e187 = idx_2;
    s_am.Store(_e187*4+_e185*16, l_u_c_vv);
    s_am.Store(0+0+0, l_u_e_ccc);
    int _e196 = idx_2;
    s_am.Store(_e196*2+0+0, l_u_e_ccv);
    int _e200 = idx_2;
    s_am.Store(0+_e200*4+0, l_u_e_cvc);
    int _e205 = idx_2;
    int _e207 = idx_2;
    s_am.Store(_e207*2+_e205*4+0, l_u_e_cvv);
    int _e210 = idx_2;
    s_am.Store(0+0+_e210*16, l_u_e_vcc);
    int _e215 = idx_2;
    int _e218 = idx_2;
    s_am.Store(_e218*2+0+_e215*16, l_u_e_vcv);
    int _e221 = idx_2;
    int _e223 = idx_2;
    s_am.Store(0+_e223*4+_e221*16, l_u_e_vvc);
    int _e227 = idx_2;
    int _e229 = idx_2;
    int _e231 = idx_2;
    s_am.Store(_e231*2+_e229*4+_e227*16, l_u_e_vvv);
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
    int idx_3 = int(1);

    int _e3 = idx_3;
    idx_3 = asint(asuint(_e3) - asuint(int(1)));
    StructWithArrayOfStructOfMat l_s_s_1 = ConstructStructWithArrayOfStructOfMat(Constructarray4_StructWithMat_(ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+0+0+0), s_sasm.Load<half2>(0+0+0+4), s_sasm.Load<half2>(0+0+0+8), s_sasm.Load<half2>(0+0+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+16+0+0), s_sasm.Load<half2>(0+16+0+4), s_sasm.Load<half2>(0+16+0+8), s_sasm.Load<half2>(0+16+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+32+0+0), s_sasm.Load<half2>(0+32+0+4), s_sasm.Load<half2>(0+32+0+8), s_sasm.Load<half2>(0+32+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+48+0+0), s_sasm.Load<half2>(0+48+0+4), s_sasm.Load<half2>(0+48+0+8), s_sasm.Load<half2>(0+48+0+12)))));
    StructWithMat l_s_a_1[4] = Constructarray4_StructWithMat_(ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+0+0+0), s_sasm.Load<half2>(0+0+0+4), s_sasm.Load<half2>(0+0+0+8), s_sasm.Load<half2>(0+0+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+16+0+0), s_sasm.Load<half2>(0+16+0+4), s_sasm.Load<half2>(0+16+0+8), s_sasm.Load<half2>(0+16+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+32+0+0), s_sasm.Load<half2>(0+32+0+4), s_sasm.Load<half2>(0+32+0+8), s_sasm.Load<half2>(0+32+0+12))), ConstructStructWithMat(half4x2(s_sasm.Load<half2>(0+48+0+0), s_sasm.Load<half2>(0+48+0+4), s_sasm.Load<half2>(0+48+0+8), s_sasm.Load<half2>(0+48+0+12))));
    half4x2 l_s_m_c_1 = half4x2(s_sasm.Load<half2>(0+0+0+0), s_sasm.Load<half2>(0+0+0+4), s_sasm.Load<half2>(0+0+0+8), s_sasm.Load<half2>(0+0+0+12));
    int _e17 = idx_3;
    half4x2 l_s_m_v_1 = half4x2(s_sasm.Load<half2>(0+_e17*16+0+0), s_sasm.Load<half2>(0+_e17*16+0+4), s_sasm.Load<half2>(0+_e17*16+0+8), s_sasm.Load<half2>(0+_e17*16+0+12));
    half2 l_s_c_cc_1 = s_sasm.Load<half2>(0+0+0+0);
    int _e31 = idx_3;
    half2 l_s_c_cv_1 = s_sasm.Load<half2>(_e31*4+0+0+0);
    int _e36 = idx_3;
    half2 l_s_c_vc_1 = s_sasm.Load<half2>(0+0+_e36*16+0);
    int _e43 = idx_3;
    int _e46 = idx_3;
    half2 l_s_c_vv_1 = s_sasm.Load<half2>(_e46*4+0+_e43*16+0);
    half l_s_e_ccc_1 = s_sasm.Load<half>(0+0+0+0+0);
    int _e61 = idx_3;
    half l_s_e_ccv_1 = s_sasm.Load<half>(_e61*2+0+0+0+0);
    int _e68 = idx_3;
    half l_s_e_cvc_1 = s_sasm.Load<half>(0+_e68*4+0+0+0);
    int _e76 = idx_3;
    int _e78 = idx_3;
    half l_s_e_cvv_1 = s_sasm.Load<half>(_e78*2+_e76*4+0+0+0);
    int _e83 = idx_3;
    half l_s_e_vcc_1 = s_sasm.Load<half>(0+0+0+_e83*16+0);
    int _e91 = idx_3;
    int _e95 = idx_3;
    half l_s_e_vcv_1 = s_sasm.Load<half>(_e95*2+0+0+_e91*16+0);
    int _e100 = idx_3;
    int _e103 = idx_3;
    half l_s_e_vvc_1 = s_sasm.Load<half>(0+_e103*4+0+_e100*16+0);
    int _e109 = idx_3;
    int _e112 = idx_3;
    int _e114 = idx_3;
    half l_s_e_vvv_1 = s_sasm.Load<half>(_e114*2+_e112*4+0+_e109*16+0);
    StructWithArrayOfStructOfMat l_u_s_1 = u_sasm;
    StructWithMat l_u_a_1[4] = u_sasm.a;
    half4x2 l_u_m_c_1 = GetMatmOnStructWithMat(u_sasm.a[0]);
    int _e129 = idx_3;
    half4x2 l_u_m_v_1 = GetMatmOnStructWithMat(u_sasm.a[_e129]);
    half2 l_u_c_cc_1 = GetMatmOnStructWithMat(u_sasm.a[0])[0];
    int _e143 = idx_3;
    half2 l_u_c_cv_1 = GetMatmOnStructWithMat(u_sasm.a[0])[_e143];
    int _e148 = idx_3;
    half2 l_u_c_vc_1 = GetMatmOnStructWithMat(u_sasm.a[_e148])[0];
    int _e155 = idx_3;
    int _e158 = idx_3;
    half2 l_u_c_vv_1 = GetMatmOnStructWithMat(u_sasm.a[_e155])[_e158];
    half l_u_e_ccc_1 = GetMatmOnStructWithMat(u_sasm.a[0])[0].x;
    int _e173 = idx_3;
    half l_u_e_ccv_1 = GetMatmOnStructWithMat(u_sasm.a[0])[0][_e173];
    int _e180 = idx_3;
    half l_u_e_cvc_1 = GetMatmOnStructWithMat(u_sasm.a[0])[_e180].x;
    int _e188 = idx_3;
    int _e190 = idx_3;
    half l_u_e_cvv_1 = GetMatmOnStructWithMat(u_sasm.a[0])[_e188][_e190];
    int _e195 = idx_3;
    half l_u_e_vcc_1 = GetMatmOnStructWithMat(u_sasm.a[_e195])[0].x;
    int _e203 = idx_3;
    int _e207 = idx_3;
    half l_u_e_vcv_1 = GetMatmOnStructWithMat(u_sasm.a[_e203])[0][_e207];
    int _e212 = idx_3;
    int _e215 = idx_3;
    half l_u_e_vvc_1 = GetMatmOnStructWithMat(u_sasm.a[_e212])[_e215].x;
    int _e221 = idx_3;
    int _e224 = idx_3;
    int _e226 = idx_3;
    half l_u_e_vvv_1 = GetMatmOnStructWithMat(u_sasm.a[_e221])[_e224][_e226];
    {
        StructWithArrayOfStructOfMat _value2 = l_u_s_1;
        {
            StructWithMat _value3[4] = _value2.a;
            {
                StructWithMat _value4 = _value3[0];
                {
                    s_sasm.Store(0+0+0+0, _value4.m_0);
                    s_sasm.Store(0+0+0+4, _value4.m_1);
                    s_sasm.Store(0+0+0+8, _value4.m_2);
                    s_sasm.Store(0+0+0+12, _value4.m_3);
                }
            }
            {
                StructWithMat _value4 = _value3[1];
                {
                    s_sasm.Store(0+16+0+0, _value4.m_0);
                    s_sasm.Store(0+16+0+4, _value4.m_1);
                    s_sasm.Store(0+16+0+8, _value4.m_2);
                    s_sasm.Store(0+16+0+12, _value4.m_3);
                }
            }
            {
                StructWithMat _value4 = _value3[2];
                {
                    s_sasm.Store(0+32+0+0, _value4.m_0);
                    s_sasm.Store(0+32+0+4, _value4.m_1);
                    s_sasm.Store(0+32+0+8, _value4.m_2);
                    s_sasm.Store(0+32+0+12, _value4.m_3);
                }
            }
            {
                StructWithMat _value4 = _value3[3];
                {
                    s_sasm.Store(0+48+0+0, _value4.m_0);
                    s_sasm.Store(0+48+0+4, _value4.m_1);
                    s_sasm.Store(0+48+0+8, _value4.m_2);
                    s_sasm.Store(0+48+0+12, _value4.m_3);
                }
            }
        }
    }
    {
        StructWithMat _value2[4] = l_u_a_1;
        {
            StructWithMat _value3 = _value2[0];
            {
                s_sasm.Store(0+0+0+0, _value3.m_0);
                s_sasm.Store(0+0+0+4, _value3.m_1);
                s_sasm.Store(0+0+0+8, _value3.m_2);
                s_sasm.Store(0+0+0+12, _value3.m_3);
            }
        }
        {
            StructWithMat _value3 = _value2[1];
            {
                s_sasm.Store(0+16+0+0, _value3.m_0);
                s_sasm.Store(0+16+0+4, _value3.m_1);
                s_sasm.Store(0+16+0+8, _value3.m_2);
                s_sasm.Store(0+16+0+12, _value3.m_3);
            }
        }
        {
            StructWithMat _value3 = _value2[2];
            {
                s_sasm.Store(0+32+0+0, _value3.m_0);
                s_sasm.Store(0+32+0+4, _value3.m_1);
                s_sasm.Store(0+32+0+8, _value3.m_2);
                s_sasm.Store(0+32+0+12, _value3.m_3);
            }
        }
        {
            StructWithMat _value3 = _value2[3];
            {
                s_sasm.Store(0+48+0+0, _value3.m_0);
                s_sasm.Store(0+48+0+4, _value3.m_1);
                s_sasm.Store(0+48+0+8, _value3.m_2);
                s_sasm.Store(0+48+0+12, _value3.m_3);
            }
        }
    }
    {
        half4x2 _value2 = l_u_m_c_1;
        s_sasm.Store(0+0+0+0, _value2[0]);
        s_sasm.Store(0+0+0+4, _value2[1]);
        s_sasm.Store(0+0+0+8, _value2[2]);
        s_sasm.Store(0+0+0+12, _value2[3]);
    }
    int _e238 = idx_3;
    {
        half4x2 _value2 = l_u_m_v_1;
        s_sasm.Store(0+_e238*16+0+0, _value2[0]);
        s_sasm.Store(0+_e238*16+0+4, _value2[1]);
        s_sasm.Store(0+_e238*16+0+8, _value2[2]);
        s_sasm.Store(0+_e238*16+0+12, _value2[3]);
    }
    s_sasm.Store(0+0+0+0, l_u_c_cc_1);
    int _e250 = idx_3;
    s_sasm.Store(_e250*4+0+0+0, l_u_c_cv_1);
    int _e254 = idx_3;
    s_sasm.Store(0+0+_e254*16+0, l_u_c_vc_1);
    int _e260 = idx_3;
    int _e263 = idx_3;
    s_sasm.Store(_e263*4+0+_e260*16+0, l_u_c_vv_1);
    s_sasm.Store(0+0+0+0+0, l_u_e_ccc_1);
    int _e276 = idx_3;
    s_sasm.Store(_e276*2+0+0+0+0, l_u_e_ccv_1);
    int _e282 = idx_3;
    s_sasm.Store(0+_e282*4+0+0+0, l_u_e_cvc_1);
    int _e289 = idx_3;
    int _e291 = idx_3;
    s_sasm.Store(_e291*2+_e289*4+0+0+0, l_u_e_cvv_1);
    int _e295 = idx_3;
    s_sasm.Store(0+0+0+_e295*16+0, l_u_e_vcc_1);
    int _e302 = idx_3;
    int _e306 = idx_3;
    s_sasm.Store(_e306*2+0+0+_e302*16+0, l_u_e_vcv_1);
    int _e310 = idx_3;
    int _e313 = idx_3;
    s_sasm.Store(0+_e313*4+0+_e310*16+0, l_u_e_vvc_1);
    int _e318 = idx_3;
    int _e321 = idx_3;
    int _e323 = idx_3;
    s_sasm.Store(_e323*2+_e321*4+0+_e318*16+0, l_u_e_vvv_1);
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
