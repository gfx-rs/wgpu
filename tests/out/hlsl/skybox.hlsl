struct NagaConstants {
    int base_vertex;
    int base_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b1);

struct VertexOutput {
    float4 position : SV_Position;
    float3 uv : LOC0;
};

struct Data {
    row_major float4x4 proj_inv;
    row_major float4x4 view;
};

cbuffer r_data : register(b0) { Data r_data; }
TextureCube<float4> r_texture : register(t0);
SamplerState r_sampler : register(s0, space1);

struct VertexOutput_vs_main {
    float3 uv : LOC0;
    float4 position : SV_Position;
};

struct FragmentInput_fs_main {
    float3 uv_1 : LOC0;
    float4 position_1 : SV_Position;
};

VertexOutput ConstructVertexOutput(float4 arg0, float3 arg1) {
    VertexOutput ret = (VertexOutput)0;
    ret.position = arg0;
    ret.uv = arg1;
    return ret;
}

VertexOutput_vs_main vs_main(uint vertex_index : SV_VertexID)
{
    int tmp1_ = (int)0;
    int tmp2_ = (int)0;

    tmp1_ = (int((_NagaConstants.base_vertex + vertex_index)) / 2);
    tmp2_ = (int((_NagaConstants.base_vertex + vertex_index)) & 1);
    int _expr9 = tmp1_;
    int _expr15 = tmp2_;
    float4 pos = float4(((float(_expr9) * 4.0) - 1.0), ((float(_expr15) * 4.0) - 1.0), 0.0, 1.0);
    float4 _expr27 = r_data.view[0];
    float4 _expr32 = r_data.view[1];
    float4 _expr37 = r_data.view[2];
    float3x3 inv_model_view = transpose(float3x3(_expr27.xyz, _expr32.xyz, _expr37.xyz));
    float4x4 _expr43 = r_data.proj_inv;
    float4 unprojected = mul(pos, _expr43);
    const VertexOutput vertexoutput = ConstructVertexOutput(pos, mul(unprojected.xyz, inv_model_view));
    const VertexOutput_vs_main vertexoutput_1 = { vertexoutput.uv, vertexoutput.position };
    return vertexoutput_1;
}

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    VertexOutput in_ = { fragmentinput_fs_main.position_1, fragmentinput_fs_main.uv_1 };
    float4 _expr4 = r_texture.Sample(r_sampler, in_.uv);
    return _expr4;
}
