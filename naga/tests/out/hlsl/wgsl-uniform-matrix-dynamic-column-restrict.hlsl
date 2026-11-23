struct S {
    float a;
    int _pad1_0;
    row_major float3x2 m;
};

struct Aligned {
    row_major float3x2 m;
    int _end_pad_0;
    int _end_pad_1;
};

struct mat3x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
    float2 c_2_;
};

struct S_std140_ {
    float a;
    int _pad1_0;
    float2 m_0_;
    float2 m_1_;
    float2 m_2_;
};

struct Aligned_std140_ {
    float2 m_0_;
    float2 m_1_;
    float2 m_2_;
    int _end_pad_0;
    int _end_pad_1;
};

cbuffer u_m : register(b0) { mat3x2f32_std140_ u_m; }
cbuffer u_s : register(b1) { S_std140_ u_s; }
cbuffer u_arr : register(b2) { Aligned_std140_ u_arr[2]; }
RWByteAddressBuffer out_ : register(u3);

[numthreads(1, 1, 1)]
void main(uint i : SV_GroupIndex)
{
    float2 _e4 = u_m.c_0_;
    float2 _e6 = u_m.c_1_;
    float2 _e8 = u_m.c_2_;
    out_.Store2(0, asuint(float3x2(_e4, _e6, _e8)[min(uint(i), 2u)]));
    float2 _e16 = u_s.m_0_;
    float2 _e18 = u_s.m_1_;
    float2 _e20 = u_s.m_2_;
    out_.Store2(8, asuint(float3x2(_e16, _e18, _e20)[min(uint(i), 2u)]));
    float2 _e27 = u_m.c_0_;
    float2 _e29 = u_m.c_1_;
    float2 _e31 = u_m.c_2_;
    float2 _e37 = u_s.m_0_;
    float2 _e39 = u_s.m_1_;
    float2 _e41 = u_s.m_2_;
    out_.Store2(16, asuint(float2(float3x2(_e27, _e29, _e31)[min(uint(i), 2u)][min(uint(i), 1u)], float3x2(_e37, _e39, _e41)[min(uint(i), 2u)][min(uint(i), 1u)])));
    float2 _e51 = u_arr[i].m_0_;
    float2 _e53 = u_arr[i].m_1_;
    float2 _e55 = u_arr[i].m_2_;
    out_.Store2(24, asuint(float3x2(_e51, _e53, _e55)[min(uint(i), 2u)]));
    float2 _e63 = u_arr[i].m_0_;
    float2 _e65 = u_arr[i].m_1_;
    float2 _e67 = u_arr[i].m_2_;
    out_.Store2(32, asuint(float2(float3x2(_e63, _e65, _e67)[min(uint(i), 2u)][min(uint(i), 1u)], 0.0)));
    return;
}
