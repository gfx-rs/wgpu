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

struct Aligned_std140_ {
    float2 m_0_;
    float2 m_1_;
    float2 m_2_;
    int _end_pad_0;
    int _end_pad_1;
};

cbuffer u_m : register(b0) { mat3x2f32_std140_ u_m; }
cbuffer u_arr : register(b1) { Aligned_std140_ u_arr[2]; }
cbuffer u_v : register(b2) { float4 u_v[2]; }
RWByteAddressBuffer out_ : register(u3);

[numthreads(1, 1, 1)]
void main(uint i : SV_GroupIndex)
{
    float2 _e4 = u_m.c_0_;
    float2 _e6 = u_m.c_1_;
    float2 _e8 = u_m.c_2_;
    out_.Store2(0, asuint(float3x2(_e4, _e6, _e8)[min(uint(i), 2u)]));
    float2 _e17 = u_arr[i].m_1_;
    out_.Store2(8, asuint(_e17));
    float2 _e23 = u_arr[i].m_0_;
    float2 _e25 = u_arr[i].m_1_;
    float2 _e27 = u_arr[i].m_2_;
    out_.Store2(16, asuint(float3x2(_e23, _e25, _e27)[min(uint(i), 2u)]));
    float4 _e34 = u_v[i];
    out_.Store2(24, asuint(_e34.xy));
    return;
}
