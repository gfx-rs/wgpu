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

[numthreads(1, 1, 1)]
void main()
{
    int idx = (int)0;
    int idx_1 = (int)0;
    int idx_2 = (int)0;
    int idx_3 = (int)0;

    idx = int(1);
    int _e3 = idx;
    idx = asint(asuint(_e3) - asuint(int(1)));
    half4x2 _e6 = ((half4x2)u_m);
    half2 _e9 = u_m._0;
    int _e11 = idx;
    half2 _e13 = __get_col_of_mat4x2_f16(u_m, _e11);
    half _e17 = u_m._0.x;
    int _e20 = idx;
    half _e22 = u_m._0[_e20];
    int _e24 = idx;
    half _e27 = __get_col_of_mat4x2_f16(u_m, _e24).x;
    int _e29 = idx;
    int _e31 = idx;
    half _e33 = __get_col_of_mat4x2_f16(u_m, _e29)[_e31];
    {
        half4x2 _value2 = _e6;
        s_m.Store(0, _value2[0]);
        s_m.Store(4, _value2[1]);
        s_m.Store(8, _value2[2]);
        s_m.Store(12, _value2[3]);
    }
    s_m.Store(0, _e9);
    int _e38 = idx;
    s_m.Store(_e38*4, _e13);
    s_m.Store(0+0, _e17);
    int _e45 = idx;
    s_m.Store(_e45*2+0, _e22);
    int _e48 = idx;
    s_m.Store(0+_e48*4, _e27);
    int _e52 = idx;
    int _e54 = idx;
    s_m.Store(_e54*2+_e52*4, _e33);
    idx_1 = int(1);
    int _e60 = idx_1;
    idx_1 = asint(asuint(_e60) - asuint(int(1)));
    StructWithMat _e63 = u_sm;
    half4x2 _e66 = GetMatmOnStructWithMat(u_sm);
    half2 _e70 = GetMatmOnStructWithMat(u_sm)[0];
    int _e73 = idx_1;
    half2 _e75 = GetMatmOnStructWithMat(u_sm)[_e73];
    half _e80 = GetMatmOnStructWithMat(u_sm)[0].x;
    int _e84 = idx_1;
    half _e86 = GetMatmOnStructWithMat(u_sm)[0][_e84];
    int _e89 = idx_1;
    half _e92 = GetMatmOnStructWithMat(u_sm)[_e89].x;
    int _e95 = idx_1;
    int _e97 = idx_1;
    half _e99 = GetMatmOnStructWithMat(u_sm)[_e95][_e97];
    {
        StructWithMat _value2 = _e63;
        {
            s_sm.Store(0+0, _value2.m_0);
            s_sm.Store(0+4, _value2.m_1);
            s_sm.Store(0+8, _value2.m_2);
            s_sm.Store(0+12, _value2.m_3);
        }
    }
    {
        half4x2 _value2 = _e66;
        s_sm.Store(0+0, _value2[0]);
        s_sm.Store(0+4, _value2[1]);
        s_sm.Store(0+8, _value2[2]);
        s_sm.Store(0+12, _value2[3]);
    }
    s_sm.Store(0+0, _e70);
    int _e108 = idx_1;
    s_sm.Store(_e108*4+0, _e75);
    s_sm.Store(0+0+0, _e80);
    int _e117 = idx_1;
    s_sm.Store(_e117*2+0+0, _e86);
    int _e121 = idx_1;
    s_sm.Store(0+_e121*4+0, _e92);
    int _e126 = idx_1;
    int _e128 = idx_1;
    s_sm.Store(_e128*2+_e126*4+0, _e99);
    idx_2 = int(1);
    int _e134 = idx_2;
    idx_2 = asint(asuint(_e134) - asuint(int(1)));
    half4x2 _e137[4] = ((half4x2[4])u_am);
    half4x2 _e140 = ((half4x2)u_am[0]);
    int _e142 = idx_2;
    half4x2 _e144 = ((half4x2)u_am[_e142]);
    half2 _e148 = u_am[0]._0;
    int _e151 = idx_2;
    half2 _e153 = __get_col_of_mat4x2_f16(u_am[0], _e151);
    int _e155 = idx_2;
    half2 _e158 = u_am[_e155]._0;
    int _e160 = idx_2;
    int _e162 = idx_2;
    half2 _e164 = __get_col_of_mat4x2_f16(u_am[_e160], _e162);
    half _e169 = u_am[0]._0.x;
    int _e173 = idx_2;
    half _e175 = u_am[0]._0[_e173];
    int _e178 = idx_2;
    half _e181 = __get_col_of_mat4x2_f16(u_am[0], _e178).x;
    int _e184 = idx_2;
    int _e186 = idx_2;
    half _e188 = __get_col_of_mat4x2_f16(u_am[0], _e184)[_e186];
    int _e190 = idx_2;
    half _e194 = u_am[_e190]._0.x;
    int _e196 = idx_2;
    int _e199 = idx_2;
    half _e201 = u_am[_e196]._0[_e199];
    int _e203 = idx_2;
    int _e205 = idx_2;
    half _e208 = __get_col_of_mat4x2_f16(u_am[_e203], _e205).x;
    int _e210 = idx_2;
    int _e212 = idx_2;
    int _e214 = idx_2;
    half _e216 = __get_col_of_mat4x2_f16(u_am[_e210], _e212)[_e214];
    {
        half4x2 _value2[4] = _e137;
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
        half4x2 _value2 = _e140;
        s_am.Store(0+0, _value2[0]);
        s_am.Store(0+4, _value2[1]);
        s_am.Store(0+8, _value2[2]);
        s_am.Store(0+12, _value2[3]);
    }
    int _e221 = idx_2;
    {
        half4x2 _value2 = _e144;
        s_am.Store(_e221*16+0, _value2[0]);
        s_am.Store(_e221*16+4, _value2[1]);
        s_am.Store(_e221*16+8, _value2[2]);
        s_am.Store(_e221*16+12, _value2[3]);
    }
    s_am.Store(0+0, _e148);
    int _e228 = idx_2;
    s_am.Store(_e228*4+0, _e153);
    int _e231 = idx_2;
    s_am.Store(0+_e231*16, _e158);
    int _e235 = idx_2;
    int _e237 = idx_2;
    s_am.Store(_e237*4+_e235*16, _e164);
    s_am.Store(0+0+0, _e169);
    int _e246 = idx_2;
    s_am.Store(_e246*2+0+0, _e175);
    int _e250 = idx_2;
    s_am.Store(0+_e250*4+0, _e181);
    int _e255 = idx_2;
    int _e257 = idx_2;
    s_am.Store(_e257*2+_e255*4+0, _e188);
    int _e260 = idx_2;
    s_am.Store(0+0+_e260*16, _e194);
    int _e265 = idx_2;
    int _e268 = idx_2;
    s_am.Store(_e268*2+0+_e265*16, _e201);
    int _e271 = idx_2;
    int _e273 = idx_2;
    s_am.Store(0+_e273*4+_e271*16, _e208);
    int _e277 = idx_2;
    int _e279 = idx_2;
    int _e281 = idx_2;
    s_am.Store(_e281*2+_e279*4+_e277*16, _e216);
    idx_3 = int(1);
    int _e287 = idx_3;
    idx_3 = asint(asuint(_e287) - asuint(int(1)));
    StructWithArrayOfStructOfMat _e290 = u_sasm;
    StructWithMat _e293[4] = u_sasm.a;
    half4x2 _e298 = GetMatmOnStructWithMat(u_sasm.a[0]);
    int _e301 = idx_3;
    half4x2 _e304 = GetMatmOnStructWithMat(u_sasm.a[_e301]);
    half2 _e310 = GetMatmOnStructWithMat(u_sasm.a[0])[0];
    int _e315 = idx_3;
    half2 _e317 = GetMatmOnStructWithMat(u_sasm.a[0])[_e315];
    int _e320 = idx_3;
    half2 _e324 = GetMatmOnStructWithMat(u_sasm.a[_e320])[0];
    int _e327 = idx_3;
    int _e330 = idx_3;
    half2 _e332 = GetMatmOnStructWithMat(u_sasm.a[_e327])[_e330];
    half _e339 = GetMatmOnStructWithMat(u_sasm.a[0])[0].x;
    int _e345 = idx_3;
    half _e347 = GetMatmOnStructWithMat(u_sasm.a[0])[0][_e345];
    int _e352 = idx_3;
    half _e355 = GetMatmOnStructWithMat(u_sasm.a[0])[_e352].x;
    int _e360 = idx_3;
    int _e362 = idx_3;
    half _e364 = GetMatmOnStructWithMat(u_sasm.a[0])[_e360][_e362];
    int _e367 = idx_3;
    half _e372 = GetMatmOnStructWithMat(u_sasm.a[_e367])[0].x;
    int _e375 = idx_3;
    int _e379 = idx_3;
    half _e381 = GetMatmOnStructWithMat(u_sasm.a[_e375])[0][_e379];
    int _e384 = idx_3;
    int _e387 = idx_3;
    half _e390 = GetMatmOnStructWithMat(u_sasm.a[_e384])[_e387].x;
    int _e393 = idx_3;
    int _e396 = idx_3;
    int _e398 = idx_3;
    half _e400 = GetMatmOnStructWithMat(u_sasm.a[_e393])[_e396][_e398];
    {
        StructWithArrayOfStructOfMat _value2 = _e290;
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
        StructWithMat _value2[4] = _e293;
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
        half4x2 _value2 = _e298;
        s_sasm.Store(0+0+0+0, _value2[0]);
        s_sasm.Store(0+0+0+4, _value2[1]);
        s_sasm.Store(0+0+0+8, _value2[2]);
        s_sasm.Store(0+0+0+12, _value2[3]);
    }
    int _e410 = idx_3;
    {
        half4x2 _value2 = _e304;
        s_sasm.Store(0+_e410*16+0+0, _value2[0]);
        s_sasm.Store(0+_e410*16+0+4, _value2[1]);
        s_sasm.Store(0+_e410*16+0+8, _value2[2]);
        s_sasm.Store(0+_e410*16+0+12, _value2[3]);
    }
    s_sasm.Store(0+0+0+0, _e310);
    int _e422 = idx_3;
    s_sasm.Store(_e422*4+0+0+0, _e317);
    int _e426 = idx_3;
    s_sasm.Store(0+0+_e426*16+0, _e324);
    int _e432 = idx_3;
    int _e435 = idx_3;
    s_sasm.Store(_e435*4+0+_e432*16+0, _e332);
    s_sasm.Store(0+0+0+0+0, _e339);
    int _e448 = idx_3;
    s_sasm.Store(_e448*2+0+0+0+0, _e347);
    int _e454 = idx_3;
    s_sasm.Store(0+_e454*4+0+0+0, _e355);
    int _e461 = idx_3;
    int _e463 = idx_3;
    s_sasm.Store(_e463*2+_e461*4+0+0+0, _e364);
    int _e467 = idx_3;
    s_sasm.Store(0+0+0+_e467*16+0, _e372);
    int _e474 = idx_3;
    int _e478 = idx_3;
    s_sasm.Store(_e478*2+0+0+_e474*16+0, _e381);
    int _e482 = idx_3;
    int _e485 = idx_3;
    s_sasm.Store(0+_e485*4+0+_e482*16+0, _e390);
    int _e490 = idx_3;
    int _e493 = idx_3;
    int _e495 = idx_3;
    s_sasm.Store(_e495*2+_e493*4+0+_e490*16+0, _e400);
    return;
}
