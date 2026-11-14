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

void SetMatm1_OnInner(Inner obj, float2x2 mat) {
    obj.m1__0 = mat[0];
    obj.m1__1 = mat[1];
}

void SetMatVecm1_OnInner(Inner obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m1__0 = vec; break; }
    case 1: { obj.m1__1 = vec; break; }
    }
}

void SetMatScalarm1_OnInner(Inner obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m1__0[vec_idx] = scalar; break; }
    case 1: { obj.m1__1[vec_idx] = scalar; break; }
    }
}

float4x2 GetMatm2_OnInner(Inner obj) {
    return float4x2(obj.m2__0, obj.m2__1, obj.m2__2, obj.m2__3);
}

void SetMatm2_OnInner(Inner obj, float4x2 mat) {
    obj.m2__0 = mat[0];
    obj.m2__1 = mat[1];
    obj.m2__2 = mat[2];
    obj.m2__3 = mat[3];
}

void SetMatVecm2_OnInner(Inner obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m2__0 = vec; break; }
    case 1: { obj.m2__1 = vec; break; }
    case 2: { obj.m2__2 = vec; break; }
    case 3: { obj.m2__3 = vec; break; }
    }
}

void SetMatScalarm2_OnInner(Inner obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m2__0[vec_idx] = scalar; break; }
    case 1: { obj.m2__1[vec_idx] = scalar; break; }
    case 2: { obj.m2__2[vec_idx] = scalar; break; }
    case 3: { obj.m2__3[vec_idx] = scalar; break; }
    }
}

[numthreads(1, 1, 1)]
void main()
{
    float2x2 k = GetMatm1_OnInner(u1_);
    float4x2 m = GetMatm2_OnInner(u2_.i);
    o.Store(0, asuint(((((m[0].x + m[1].x) + m[2].x) + m[3].x) + k[0].x)));
    return;
}
