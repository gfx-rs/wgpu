struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct ImmediateDataVert {
    float position_clip;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    row_major float3x3 matrix_;
    int _end_pad_0;
};

struct ImmediateDataFrag {
    float multiplier;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    float4 tint;
};

struct FragmentIn {
    float4 color : LOC0;
};

ConstantBuffer<ImmediateDataVert> im_vert: register(b0);
ConstantBuffer<ImmediateDataFrag> im_frag: register(b0);

struct FragmentInput_main {
    float4 color : LOC0;
};

float4 vert_main(float2 pos : LOC0, uint ii : SV_InstanceID, uint vi : SV_VertexID) : SV_Position
{
    float _e9 = im_vert.position_clip;
    return float4(((float((_NagaConstants.first_instance + ii)) * float((_NagaConstants.first_vertex + vi))) * pos), 0.0, _e9);
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn in_ = { fragmentinput_main.color };
    float4 _e4 = im_frag.tint;
    return (in_.color * _e4);
}
