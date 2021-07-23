
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

struct FragmentInput_main {
    FragmentInput val1;
};

FragmentInput main()
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
    const FragmentInput fragmentinput1 = _expr30;
    return fragmentinput1;
}

void main1(FragmentInput_main fragmentinput_main)
{
    return;
}
