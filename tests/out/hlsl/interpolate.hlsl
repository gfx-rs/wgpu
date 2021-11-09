
struct FragmentInput {
    float4 position : SV_Position;
    nointerpolation uint flat : LOC0;
    noperspective float linear_ : LOC1;
    noperspective centroid float2 linear_centroid : LOC2;
    noperspective sample float3 linear_sample : LOC3;
    linear float4 perspective : LOC4;
    linear centroid float perspective_centroid : LOC5;
    linear sample float perspective_sample : LOC6;
};

struct VertexOutput_main {
    uint flat : LOC0;
    float linear_ : LOC1;
    float2 linear_centroid : LOC2;
    float3 linear_sample : LOC3;
    float4 perspective : LOC4;
    float perspective_centroid : LOC5;
    float perspective_sample : LOC6;
    float4 position : SV_Position;
};

struct FragmentInput_main {
    uint flat_1 : LOC0;
    float linear_1 : LOC1;
    float2 linear_centroid_1 : LOC2;
    float3 linear_sample_1 : LOC3;
    float4 perspective_1 : LOC4;
    float perspective_centroid_1 : LOC5;
    float perspective_sample_1 : LOC6;
    float4 position_1 : SV_Position;
};

VertexOutput_main main()
{
    FragmentInput out_ = (FragmentInput)0;

    out_.position = float4(2.0, 4.0, 5.0, 6.0);
    out_.flat = 8u;
    out_.linear_ = 27.0;
    out_.linear_centroid = float2(64.0, 125.0);
    out_.linear_sample = float3(216.0, 343.0, 512.0);
    out_.perspective = float4(729.0, 1000.0, 1331.0, 1728.0);
    out_.perspective_centroid = 2197.0;
    out_.perspective_sample = 2744.0;
    FragmentInput _expr30 = out_;
    const FragmentInput fragmentinput = _expr30;
    const VertexOutput_main fragmentinput_1 = { fragmentinput.flat, fragmentinput.linear_, fragmentinput.linear_centroid, fragmentinput.linear_sample, fragmentinput.perspective, fragmentinput.perspective_centroid, fragmentinput.perspective_sample, fragmentinput.position };
    return fragmentinput_1;
}

void main_1(FragmentInput_main fragmentinput_main)
{
    FragmentInput val = { fragmentinput_main.position_1, fragmentinput_main.flat_1, fragmentinput_main.linear_1, fragmentinput_main.linear_centroid_1, fragmentinput_main.linear_sample_1, fragmentinput_main.perspective_1, fragmentinput_main.perspective_centroid_1, fragmentinput_main.perspective_sample_1 };
    return;
}
