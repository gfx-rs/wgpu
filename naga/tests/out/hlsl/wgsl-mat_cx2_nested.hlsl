struct Inner {
    row_major float2x2 m1_;
    row_major float4x2 m2_;
};

struct Outer {
    float4 a;
    Inner i;
};

struct Inner_std140_ {
    float2 m1_0_;
    float2 m1_1_;
    float2 m2_0_;
    float2 m2_1_;
    float2 m2_2_;
    float2 m2_3_;
};

struct Outer_std140_ {
    float4 a;
    Inner_std140_ i;
};

cbuffer u1_ : register(b0) { Inner_std140_ u1_; }
cbuffer u2_ : register(b1) { Outer_std140_ u2_; }
RWByteAddressBuffer o : register(u2);

[numthreads(1, 1, 1)]
void main()
{
    float2 _e2 = u1_.m1_0_;
    float2 _e4 = u1_.m1_1_;
    float2x2 k = float2x2(_e2, _e4);
    float2 _e9 = u2_.i.m2_0_;
    float2 _e11 = u2_.i.m2_1_;
    float2 _e13 = u2_.i.m2_2_;
    float2 _e15 = u2_.i.m2_3_;
    float4x2 m = float4x2(_e9, _e11, _e13, _e15);
    o.Store(0, asuint(((((m[0].x + m[1].x) + m[2].x) + m[3].x) + k[0].x)));
    return;
}
