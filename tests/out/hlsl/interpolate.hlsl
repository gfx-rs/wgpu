
struct FragmentInput {
    float4 position : SV_Position;
    nointerpolation uint flat : LOC0;
    noperspective float linear1 : LOC1;
    noperspective centroid float2 linear_centroid : LOC2;
    noperspective sample float3 linear_sample : LOC3;
    linear float4 perspective : LOC4;
    linear centroid float perspective_centroid : LOC5;
    linear sample float perspective_sample : LOC6;
};

struct VertexOutput_main {
    uint flat : LOC0;
    float linear1 : LOC1;
    float2 linear_centroid : LOC2;
    float3 linear_sample : LOC3;
    float4 perspective : LOC4;
    float perspective_centroid : LOC5;
    float perspective_sample : LOC6;
    float4 position : SV_Position;
};

struct FragmentInput_main {
    uint flat1 : LOC0;
    float linear2 : LOC1;
    float2 linear_centroid1 : LOC2;
    float3 linear_sample1 : LOC3;
    float4 perspective1 : LOC4;
    float perspective_centroid1 : LOC5;
    float perspective_sample1 : LOC6;
    float4 position1 : SV_Position;
};

VertexOutput_main main()
{
    FragmentInput out1 = (FragmentInput)0;

    out1.position = float4(2.0, 4.0, 5.0, 6.0);
    out1.flat = 8u;
    out1.linear1 = 27.0;
    out1.linear_centroid = float2(64.0, 125.0);
    out1.linear_sample = float3(216.0, 343.0, 512.0);
    out1.perspective = float4(729.0, 1000.0, 1331.0, 1728.0);
    out1.perspective_centroid = 2197.0;
    out1.perspective_sample = 2744.0;
    FragmentInput _expr30 = out1;
    const FragmentInput fragmentinput = _expr30;
    const VertexOutput_main fragmentinput1 = { fragmentinput.flat, fragmentinput.linear1, fragmentinput.linear_centroid, fragmentinput.linear_sample, fragmentinput.perspective, fragmentinput.perspective_centroid, fragmentinput.perspective_sample, fragmentinput.position };
    return fragmentinput1;
}

void main1(FragmentInput_main fragmentinput_main)
{
    FragmentInput val = { fragmentinput_main.position1, fragmentinput_main.flat1, fragmentinput_main.linear2, fragmentinput_main.linear_centroid1, fragmentinput_main.linear_sample1, fragmentinput_main.perspective1, fragmentinput_main.perspective_centroid1, fragmentinput_main.perspective_sample1 };
    return;
}
