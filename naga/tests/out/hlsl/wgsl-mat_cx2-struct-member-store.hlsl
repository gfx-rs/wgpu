struct R {
    row_major float2x2 M;
    row_major float2x2 Minv;
};

struct Wide {
    row_major float4x2 d;
    row_major float3x2 b;
};

struct Outer {
    R inner;
    Wide arr[2];
};

struct WithMatArray {
    row_major float4x2 am[2];
};

RWByteAddressBuffer o : register(u0);
static R p = (R)0;
groupshared Wide w;

R ZeroValueR() {
    return (R)0;
}

Wide ZeroValueWide() {
    return (Wide)0;
}

Outer ZeroValueOuter() {
    return (Outer)0;
}

float ZeroValuefloat() {
    return (float)0;
}

WithMatArray ZeroValueWithMatArray() {
    return (WithMatArray)0;
}

[numthreads(1, 1, 1)]
void main(uint li : SV_GroupIndex)
{
    float a = (float)0;
    R r = (R)0;
    R local = (R)0;
    int idx = (int)0;
    Wide t = (Wide)0;
    Wide local_1 = (Wide)0;
    int idx_1 = (int)0;
    Outer n = (Outer)0;
    float x = (float)0;
    Outer local_2 = (Outer)0;
    int idx_2 = (int)0;
    WithMatArray m = (WithMatArray)0;
    WithMatArray local_3 = (WithMatArray)0;

    if ((li == 0u)) {
        w = ZeroValueWide();
    }
    GroupMemoryBarrierWithGroupSync();
    int idx_3 = int(li);
    a = 0.5;
    float _e6 = a;
    r = ZeroValueR();
    r.M = float2x2(float2(1.0, 2.0), float2(3.0, _e6));
    r.Minv = float2x2(float2(5.0, 6.0), float2(7.0, 8.0));
    R _e23 = r;
    local = _e23;
    R r_1 = local;
    o.Store(0, asuint(r_1.M[0].x));
    o.Store(4, asuint(r_1.Minv[1].y));
    idx = idx_3;
    int _e37 = idx;
    t = ZeroValueWide();
    t.d = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
    t.b = float3x2(float2(9.0, 10.0), float2(11.0, 12.0), float2(13.0, 14.0));
    t.d[min(uint(_e37), 3u)] = float2(20.0, 21.0);
    t.b[min(uint(_e37), 2u)].y = 30.0;
    t.b[min(uint(_e37), 2u)][min(uint(_e37), 1u)] = 31.0;
    Wide _e75 = t;
    local_1 = _e75;
    Wide t_1 = local_1;
    o.Store(8, asuint(t_1.d[0].x));
    o.Store(12, asuint(t_1.b[0].y));
    idx_1 = idx_3;
    int _e89 = idx_1;
    n = ZeroValueOuter();
    x = ZeroValuefloat();
    n.inner.M = float2x2(float2(1.0, 2.0), float2(3.0, 4.0));
    n.inner.Minv[min(uint(_e89), 1u)] = float2(5.0, 6.0);
    n.arr[min(uint(_e89), 1u)].d = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
    n.arr[1].b[min(uint(_e89), 2u)] = float2(9.0, 10.0);
    n.arr[0].b[min(uint(_e89), 2u)][min(uint(_e89), 1u)] = 11.0;
    x = 12.0;
    float _e136 = x;
    n.inner.M = float2x2(float2(_e136, _e136), float2(_e136, _e136));
    Outer _e141 = n;
    local_2 = _e141;
    Outer n_1 = local_2;
    o.Store(16, asuint(n_1.inner.M[0].x));
    o.Store(20, asuint(n_1.arr[0].b[0].x));
    idx_2 = idx_3;
    int _e158 = idx_2;
    m = ZeroValueWithMatArray();
    m.am[0][min(uint(_e158), 3u)] = float2(1.0, 2.0);
    m.am[1][min(uint(_e158), 3u)][min(uint(_e158), 1u)] = 3.0;
    WithMatArray _e170 = m;
    local_3 = _e170;
    WithMatArray m_1 = local_3;
    o.Store(24, asuint(m_1.am[1][0].x));
    p.M = float2x2(float2(1.0, 2.0), float2(3.0, 4.0));
    p.Minv[min(uint(idx_3), 1u)] = float2(5.0, 6.0);
    float _e200 = p.M[0].x;
    float _e205 = p.Minv[0].x;
    o.Store(28, asuint((_e200 + _e205)));
    w.d = float4x2(float2(1.0, 2.0), float2(3.0, 4.0), float2(5.0, 6.0), float2(7.0, 8.0));
    w.b[min(uint(idx_3), 2u)][min(uint(idx_3), 1u)] = 9.0;
    GroupMemoryBarrierWithGroupSync();
    float _e232 = w.d[0].x;
    float _e237 = w.b[0].x;
    o.Store(32, asuint((_e232 + _e237)));
    return;
}
