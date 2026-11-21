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
RWByteAddressBuffer s_am : register(u0, space2);
cbuffer u_am : register(b1, space2) { row_major float3x3 u_am[4]; }
RWByteAddressBuffer s_sasm : register(u0, space3);
cbuffer u_sasm : register(b1, space3) { StructWithArrayOfStructOfMat u_sasm; }

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
    float3x3 _e6 = u_m;
    float3 _e9 = u_m[0];
    int _e11 = idx;
    float3 _e13 = u_m[_e11];
    float _e17 = u_m[0].x;
    int _e20 = idx;
    float _e22 = u_m[0][_e20];
    int _e24 = idx;
    float _e27 = u_m[_e24].x;
    int _e29 = idx;
    int _e31 = idx;
    float _e33 = u_m[_e29][_e31];
    {
        float3x3 _value2 = _e6;
        s_m.Store3(0, asuint(_value2[0]));
        s_m.Store3(16, asuint(_value2[1]));
        s_m.Store3(32, asuint(_value2[2]));
    }
    s_m.Store3(0, asuint(_e9));
    int _e38 = idx;
    s_m.Store3(_e38*16, asuint(_e13));
    s_m.Store(0+0, asuint(_e17));
    int _e45 = idx;
    s_m.Store(_e45*4+0, asuint(_e22));
    int _e48 = idx;
    s_m.Store(0+_e48*16, asuint(_e27));
    int _e52 = idx;
    int _e54 = idx;
    s_m.Store(_e54*4+_e52*16, asuint(_e33));
    idx_1 = int(1);
    int _e60 = idx_1;
    idx_1 = asint(asuint(_e60) - asuint(int(1)));
    StructWithMat _e63 = u_sm;
    float3x3 _e66 = u_sm.m;
    float3 _e70 = u_sm.m[0];
    int _e73 = idx_1;
    float3 _e75 = u_sm.m[_e73];
    float _e80 = u_sm.m[0].x;
    int _e84 = idx_1;
    float _e86 = u_sm.m[0][_e84];
    int _e89 = idx_1;
    float _e92 = u_sm.m[_e89].x;
    int _e95 = idx_1;
    int _e97 = idx_1;
    float _e99 = u_sm.m[_e95][_e97];
    {
        StructWithMat _value2 = _e63;
        {
            float3x3 _value3 = _value2.m;
            s_sm.Store3(0+0, asuint(_value3[0]));
            s_sm.Store3(0+16, asuint(_value3[1]));
            s_sm.Store3(0+32, asuint(_value3[2]));
        }
    }
    {
        float3x3 _value2 = _e66;
        s_sm.Store3(0+0, asuint(_value2[0]));
        s_sm.Store3(0+16, asuint(_value2[1]));
        s_sm.Store3(0+32, asuint(_value2[2]));
    }
    s_sm.Store3(0+0, asuint(_e70));
    int _e108 = idx_1;
    s_sm.Store3(_e108*16+0, asuint(_e75));
    s_sm.Store(0+0+0, asuint(_e80));
    int _e117 = idx_1;
    s_sm.Store(_e117*4+0+0, asuint(_e86));
    int _e121 = idx_1;
    s_sm.Store(0+_e121*16+0, asuint(_e92));
    int _e126 = idx_1;
    int _e128 = idx_1;
    s_sm.Store(_e128*4+_e126*16+0, asuint(_e99));
    idx_2 = int(1);
    int _e134 = idx_2;
    idx_2 = asint(asuint(_e134) - asuint(int(1)));
    float3x3 _e137[4] = u_am;
    float3x3 _e140 = u_am[0];
    int _e142 = idx_2;
    float3x3 _e144 = u_am[_e142];
    float3 _e148 = u_am[0][0];
    int _e151 = idx_2;
    float3 _e153 = u_am[0][_e151];
    int _e155 = idx_2;
    float3 _e158 = u_am[_e155][0];
    int _e160 = idx_2;
    int _e162 = idx_2;
    float3 _e164 = u_am[_e160][_e162];
    float _e169 = u_am[0][0].x;
    int _e173 = idx_2;
    float _e175 = u_am[0][0][_e173];
    int _e178 = idx_2;
    float _e181 = u_am[0][_e178].x;
    int _e184 = idx_2;
    int _e186 = idx_2;
    float _e188 = u_am[0][_e184][_e186];
    int _e190 = idx_2;
    float _e194 = u_am[_e190][0].x;
    int _e196 = idx_2;
    int _e199 = idx_2;
    float _e201 = u_am[_e196][0][_e199];
    int _e203 = idx_2;
    int _e205 = idx_2;
    float _e208 = u_am[_e203][_e205].x;
    int _e210 = idx_2;
    int _e212 = idx_2;
    int _e214 = idx_2;
    float _e216 = u_am[_e210][_e212][_e214];
    {
        float3x3 _value2[4] = _e137;
        {
            float3x3 _value3 = _value2[0];
            s_am.Store3(0+0, asuint(_value3[0]));
            s_am.Store3(0+16, asuint(_value3[1]));
            s_am.Store3(0+32, asuint(_value3[2]));
        }
        {
            float3x3 _value3 = _value2[1];
            s_am.Store3(48+0, asuint(_value3[0]));
            s_am.Store3(48+16, asuint(_value3[1]));
            s_am.Store3(48+32, asuint(_value3[2]));
        }
        {
            float3x3 _value3 = _value2[2];
            s_am.Store3(96+0, asuint(_value3[0]));
            s_am.Store3(96+16, asuint(_value3[1]));
            s_am.Store3(96+32, asuint(_value3[2]));
        }
        {
            float3x3 _value3 = _value2[3];
            s_am.Store3(144+0, asuint(_value3[0]));
            s_am.Store3(144+16, asuint(_value3[1]));
            s_am.Store3(144+32, asuint(_value3[2]));
        }
    }
    {
        float3x3 _value2 = _e140;
        s_am.Store3(0+0, asuint(_value2[0]));
        s_am.Store3(0+16, asuint(_value2[1]));
        s_am.Store3(0+32, asuint(_value2[2]));
    }
    int _e221 = idx_2;
    {
        float3x3 _value2 = _e144;
        s_am.Store3(_e221*48+0, asuint(_value2[0]));
        s_am.Store3(_e221*48+16, asuint(_value2[1]));
        s_am.Store3(_e221*48+32, asuint(_value2[2]));
    }
    s_am.Store3(0+0, asuint(_e148));
    int _e228 = idx_2;
    s_am.Store3(_e228*16+0, asuint(_e153));
    int _e231 = idx_2;
    s_am.Store3(0+_e231*48, asuint(_e158));
    int _e235 = idx_2;
    int _e237 = idx_2;
    s_am.Store3(_e237*16+_e235*48, asuint(_e164));
    s_am.Store(0+0+0, asuint(_e169));
    int _e246 = idx_2;
    s_am.Store(_e246*4+0+0, asuint(_e175));
    int _e250 = idx_2;
    s_am.Store(0+_e250*16+0, asuint(_e181));
    int _e255 = idx_2;
    int _e257 = idx_2;
    s_am.Store(_e257*4+_e255*16+0, asuint(_e188));
    int _e260 = idx_2;
    s_am.Store(0+0+_e260*48, asuint(_e194));
    int _e265 = idx_2;
    int _e268 = idx_2;
    s_am.Store(_e268*4+0+_e265*48, asuint(_e201));
    int _e271 = idx_2;
    int _e273 = idx_2;
    s_am.Store(0+_e273*16+_e271*48, asuint(_e208));
    int _e277 = idx_2;
    int _e279 = idx_2;
    int _e281 = idx_2;
    s_am.Store(_e281*4+_e279*16+_e277*48, asuint(_e216));
    idx_3 = int(1);
    int _e287 = idx_3;
    idx_3 = asint(asuint(_e287) - asuint(int(1)));
    StructWithArrayOfStructOfMat _e290 = u_sasm;
    StructWithMat _e293[4] = u_sasm.a;
    float3x3 _e298 = u_sasm.a[0].m;
    int _e301 = idx_3;
    float3x3 _e304 = u_sasm.a[_e301].m;
    float3 _e310 = u_sasm.a[0].m[0];
    int _e315 = idx_3;
    float3 _e317 = u_sasm.a[0].m[_e315];
    int _e320 = idx_3;
    float3 _e324 = u_sasm.a[_e320].m[0];
    int _e327 = idx_3;
    int _e330 = idx_3;
    float3 _e332 = u_sasm.a[_e327].m[_e330];
    float _e339 = u_sasm.a[0].m[0].x;
    int _e345 = idx_3;
    float _e347 = u_sasm.a[0].m[0][_e345];
    int _e352 = idx_3;
    float _e355 = u_sasm.a[0].m[_e352].x;
    int _e360 = idx_3;
    int _e362 = idx_3;
    float _e364 = u_sasm.a[0].m[_e360][_e362];
    int _e367 = idx_3;
    float _e372 = u_sasm.a[_e367].m[0].x;
    int _e375 = idx_3;
    int _e379 = idx_3;
    float _e381 = u_sasm.a[_e375].m[0][_e379];
    int _e384 = idx_3;
    int _e387 = idx_3;
    float _e390 = u_sasm.a[_e384].m[_e387].x;
    int _e393 = idx_3;
    int _e396 = idx_3;
    int _e398 = idx_3;
    float _e400 = u_sasm.a[_e393].m[_e396][_e398];
    {
        StructWithArrayOfStructOfMat _value2 = _e290;
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
        StructWithMat _value2[4] = _e293;
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
        float3x3 _value2 = _e298;
        s_sasm.Store3(0+0+0+0, asuint(_value2[0]));
        s_sasm.Store3(0+0+0+16, asuint(_value2[1]));
        s_sasm.Store3(0+0+0+32, asuint(_value2[2]));
    }
    int _e410 = idx_3;
    {
        float3x3 _value2 = _e304;
        s_sasm.Store3(0+_e410*48+0+0, asuint(_value2[0]));
        s_sasm.Store3(0+_e410*48+0+16, asuint(_value2[1]));
        s_sasm.Store3(0+_e410*48+0+32, asuint(_value2[2]));
    }
    s_sasm.Store3(0+0+0+0, asuint(_e310));
    int _e422 = idx_3;
    s_sasm.Store3(_e422*16+0+0+0, asuint(_e317));
    int _e426 = idx_3;
    s_sasm.Store3(0+0+_e426*48+0, asuint(_e324));
    int _e432 = idx_3;
    int _e435 = idx_3;
    s_sasm.Store3(_e435*16+0+_e432*48+0, asuint(_e332));
    s_sasm.Store(0+0+0+0+0, asuint(_e339));
    int _e448 = idx_3;
    s_sasm.Store(_e448*4+0+0+0+0, asuint(_e347));
    int _e454 = idx_3;
    s_sasm.Store(0+_e454*16+0+0+0, asuint(_e355));
    int _e461 = idx_3;
    int _e463 = idx_3;
    s_sasm.Store(_e463*4+_e461*16+0+0+0, asuint(_e364));
    int _e467 = idx_3;
    s_sasm.Store(0+0+0+_e467*48+0, asuint(_e372));
    int _e474 = idx_3;
    int _e478 = idx_3;
    s_sasm.Store(_e478*4+0+0+_e474*48+0, asuint(_e381));
    int _e482 = idx_3;
    int _e485 = idx_3;
    s_sasm.Store(0+_e485*16+0+_e482*48+0, asuint(_e390));
    int _e490 = idx_3;
    int _e493 = idx_3;
    int _e495 = idx_3;
    s_sasm.Store(_e495*4+_e493*16+0+_e490*48+0, asuint(_e400));
    return;
}
