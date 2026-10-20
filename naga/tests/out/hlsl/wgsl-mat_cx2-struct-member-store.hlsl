typedef struct { float2 _0; float2 _1; float2 _2; float2 _3; } __mat4x2_f32;
float2 __get_col_of_mat4x2_f32(__mat4x2_f32 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    case 3: { return mat._3; }
    default: { return (float2)0; }
    }
}

struct R {
    float2 M_0; float2 M_1;
    float2 Minv_0; float2 Minv_1;
};

struct Wide {
    float2 a_0; float2 a_1; float2 a_2; float2 a_3;
    float2 b_0; float2 b_1; float2 b_2;
};

struct Outer {
    R inner;
    Wide arr[2];
};

struct WithMatArray {
    __mat4x2_f32 am[2];
};

RWByteAddressBuffer o : register(u0);
static R p = (R)0;
groupshared Wide w;

float2x2 GetMatMOnR(R obj) {
    return float2x2(obj.M_0, obj.M_1);
}

float2x2 GetMatMinvOnR(R obj) {
    return float2x2(obj.Minv_0, obj.Minv_1);
}

R make(float a)
{
    R r = (R)0;

    {
        float2x2 mat = float2x2(float2(1.0, 2.0), float2(3.0, a));
        r.M_0 = mat[0];
        r.M_1 = mat[1];
    }
    {
        float2x2 mat_1 = float2x2(float2(5.0, 6.0), float2(7.0, 8.0));
        r.Minv_0 = mat_1[0];
        r.Minv_1 = mat_1[1];
    }
    R _e17 = r;
    const R r_2 = _e17;
    return r_2;
}

void store_through_pointer(inout R r_1, float x)
{
    {
        float2x2 mat_2 = float2x2(float2(x, x), float2(x, x));
        r_1.M_0 = mat_2[0];
        r_1.M_1 = mat_2[1];
    }
    return;
}

float4x2 GetMataOnWide(Wide obj) {
    return float4x2(obj.a_0, obj.a_1, obj.a_2, obj.a_3);
}

float3x2 GetMatbOnWide(Wide obj) {
    return float3x2(obj.b_0, obj.b_1, obj.b_2);
}

Wide local_stores(int idx)
{
    Wide t = (Wide)0;

    {
        float4x2 mat_3 = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
        t.a_0 = mat_3[0];
        t.a_1 = mat_3[1];
        t.a_2 = mat_3[2];
        t.a_3 = mat_3[3];
    }
    {
        float3x2 mat_4 = float3x2(float2(9.0, 10.0), float2(11.0, 12.0), float2(13.0, 14.0));
        t.b_0 = mat_4[0];
        t.b_1 = mat_4[1];
        t.b_2 = mat_4[2];
    }
    switch(idx) {
    case 0: { t.a_0 = float2(20.0, 21.0); break; }
    case 1: { t.a_1 = float2(20.0, 21.0); break; }
    case 2: { t.a_2 = float2(20.0, 21.0); break; }
    case 3: { t.a_3 = float2(20.0, 21.0); break; }
    }
    switch(idx) {
    case 0: { t.b_0[1] = 30.0; break; }
    case 1: { t.b_1[1] = 30.0; break; }
    case 2: { t.b_2[1] = 30.0; break; }
    }
    switch(idx) {
    case 0: { t.b_0[idx] = 31.0; break; }
    case 1: { t.b_1[idx] = 31.0; break; }
    case 2: { t.b_2[idx] = 31.0; break; }
    }
    Wide _e40 = t;
    const Wide wide = _e40;
    return wide;
}

Outer nested_stores(int idx_1)
{
    Outer n = (Outer)0;

    {
        float2x2 mat_5 = float2x2(float2(1.0, 2.0), float2(3.0, 4.0));
        n.inner.M_0 = mat_5[0];
        n.inner.M_1 = mat_5[1];
    }
    switch(idx_1) {
    case 0: { n.inner.Minv_0 = float2(5.0, 6.0); break; }
    case 1: { n.inner.Minv_1 = float2(5.0, 6.0); break; }
    }
    {
        float4x2 mat_6 = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
        n.arr[min(uint(idx_1), 1u)].a_0 = mat_6[0];
        n.arr[min(uint(idx_1), 1u)].a_1 = mat_6[1];
        n.arr[min(uint(idx_1), 1u)].a_2 = mat_6[2];
        n.arr[min(uint(idx_1), 1u)].a_3 = mat_6[3];
    }
    switch(idx_1) {
    case 0: { n.arr[1].b_0 = float2(9.0, 10.0); break; }
    case 1: { n.arr[1].b_1 = float2(9.0, 10.0); break; }
    case 2: { n.arr[1].b_2 = float2(9.0, 10.0); break; }
    }
    switch(idx_1) {
    case 0: { n.arr[0].b_0[idx_1] = 11.0; break; }
    case 1: { n.arr[0].b_1[idx_1] = 11.0; break; }
    case 2: { n.arr[0].b_2[idx_1] = 11.0; break; }
    }
    store_through_pointer(n.inner, 12.0);
    Outer _e48 = n;
    const Outer outer = _e48;
    return outer;
}

WithMatArray mat_array_stores(int idx_2)
{
    WithMatArray m = (WithMatArray)0;

    switch(idx_2) {
    case 0: { m.am[0]._0 = float2(1.0, 2.0); break; }
    case 1: { m.am[0]._1 = float2(1.0, 2.0); break; }
    case 2: { m.am[0]._2 = float2(1.0, 2.0); break; }
    case 3: { m.am[0]._3 = float2(1.0, 2.0); break; }
    }
    switch(idx_2) {
    case 0: { m.am[1]._0[idx_2] = 3.0; break; }
    case 1: { m.am[1]._1[idx_2] = 3.0; break; }
    case 2: { m.am[1]._2[idx_2] = 3.0; break; }
    case 3: { m.am[1]._3[idx_2] = 3.0; break; }
    }
    WithMatArray _e13 = m;
    const WithMatArray withmatarray = _e13;
    return withmatarray;
}

[numthreads(1, 1, 1)]
void main(uint li : SV_GroupIndex)
{
    if (li == 0) {
        w = (Wide)0;
    }
    GroupMemoryBarrierWithGroupSync();
    int idx_3 = int(li);
    const R _e3 = make(0.5);
    o.Store(0, asuint(GetMatMOnR(_e3)[0].x));
    o.Store(4, asuint(GetMatMinvOnR(_e3)[1].y));
    const Wide _e14 = local_stores(idx_3);
    o.Store(8, asuint(GetMataOnWide(_e14)[0].x));
    o.Store(12, asuint(GetMatbOnWide(_e14)[0].y));
    const Outer _e25 = nested_stores(idx_3);
    o.Store(16, asuint(GetMatMOnR(_e25.inner)[0].x));
    o.Store(20, asuint(GetMatbOnWide(_e25.arr[0])[0].x));
    const WithMatArray _e39 = mat_array_stores(idx_3);
    o.Store(24, asuint(_e39.am[1]._0.x));
    {
        float2x2 mat_7 = float2x2(float2(1.0, 2.0), float2(3.0, 4.0));
        p.M_0 = mat_7[0];
        p.M_1 = mat_7[1];
    }
    switch(idx_3) {
    case 0: { p.Minv_0 = float2(5.0, 6.0); break; }
    case 1: { p.Minv_1 = float2(5.0, 6.0); break; }
    }
    float _e67 = GetMatMOnR(p)[0].x;
    float _e72 = GetMatMinvOnR(p)[0].x;
    o.Store(28, asuint((_e67 + _e72)));
    {
        float4x2 mat_8 = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
        w.a_0 = mat_8[0];
        w.a_1 = mat_8[1];
        w.a_2 = mat_8[2];
        w.a_3 = mat_8[3];
    }
    switch(idx_3) {
    case 0: { w.b_0[idx_3] = 9.0; break; }
    case 1: { w.b_1[idx_3] = 9.0; break; }
    case 2: { w.b_2[idx_3] = 9.0; break; }
    }
    GroupMemoryBarrierWithGroupSync();
    float _e100 = GetMataOnWide(w)[0].x;
    float _e105 = GetMatbOnWide(w)[0].x;
    o.Store(32, asuint((_e100 + _e105)));
    return;
}
