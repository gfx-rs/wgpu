struct NagaConstants {
    int base_vertex;
    int base_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct PushConstants {
    float multiplier;
};

struct FragmentIn {
    float4 color : LOC0;
};

ConstantBuffer<PushConstants> pc: register(b0);

struct FragmentInput_main {
    float4 color : LOC0;
};

float4 vert_main(float2 pos : LOC0, uint vi : SV_VertexID) : SV_Position
{
    float _expr5 = pc.multiplier;
    return float4(((float((_NagaConstants.base_vertex + vi)) * _expr5) * pos), 0.0, 1.0);
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn in_ = { fragmentinput_main.color };
    float _expr4 = pc.multiplier;
    return (in_.color * _expr4);
}
