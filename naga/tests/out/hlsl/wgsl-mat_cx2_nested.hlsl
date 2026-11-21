struct Inner {
    float2 m1__0; float2 m1__1;
    float2 m2__0; float2 m2__1; float2 m2__2; float2 m2__3;
};

struct Outer {
    float4 a;
    Inner i;
};

cbuffer u1_ : register(b0) { Inner u1_; }
cbuffer u2_ : register(b1) { Outer u2_; }
RWByteAddressBuffer o : register(u2);

float2x2 GetMatm1_OnInner(Inner obj) {
    return float2x2(obj.m1__0, obj.m1__1);
}

float4x2 GetMatm2_OnInner(Inner obj) {
    return float4x2(obj.m2__0, obj.m2__1, obj.m2__2, obj.m2__3);
}

[numthreads(1, 1, 1)]
void main()
{
    float2x2 k = GetMatm1_OnInner(u1_);
    float4x2 m = GetMatm2_OnInner(u2_.i);
    o.Store(0, asuint(((((m[0].x + m[1].x) + m[2].x) + m[3].x) + k[0].x)));
    return;
}
