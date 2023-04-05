struct FragmentInput {
    float4 position : SV_Position;
    nointerpolation uint _flat : LOC0;
    noperspective float _linear : LOC1;
    noperspective centroid float2 linear_centroid : LOC2;
    noperspective sample float3 linear_sample : LOC3;
    float4 perspective : LOC4;
    centroid float perspective_centroid : LOC5;
    sample float perspective_sample : LOC6;
};

struct VertexOutput_vert_main {
    nointerpolation uint _flat : LOC0;
    noperspective float _linear : LOC1;
    noperspective centroid float2 linear_centroid : LOC2;
    noperspective sample float3 linear_sample : LOC3;
    float4 perspective : LOC4;
    centroid float perspective_centroid : LOC5;
    sample float perspective_sample : LOC6;
    float4 position : SV_Position;
};

struct FragmentInput_frag_main {
    nointerpolation uint _flat_1 : LOC0;
    noperspective float _linear_1 : LOC1;
    noperspective centroid float2 linear_centroid_1 : LOC2;
    noperspective sample float3 linear_sample_1 : LOC3;
    float4 perspective_1 : LOC4;
    centroid float perspective_centroid_1 : LOC5;
    sample float perspective_sample_1 : LOC6;
    float4 position_1 : SV_Position;
};

VertexOutput_vert_main vert_main()
{
    FragmentInput out_ = (FragmentInput)0;

    out_.position = float4(2.0, 4.0, 5.0, 6.0);
    out_._flat = 8u;
    out_._linear = 27.0;
    out_.linear_centroid = float2(64.0, 125.0);
    out_.linear_sample = float3(216.0, 343.0, 512.0);
    out_.perspective = float4(729.0, 1000.0, 1331.0, 1728.0);
    out_.perspective_centroid = 2197.0;
    out_.perspective_sample = 2744.0;
    FragmentInput _expr30 = out_;
    const FragmentInput fragmentinput = _expr30;
    const VertexOutput_vert_main fragmentinput_1 = { fragmentinput._flat, fragmentinput._linear, fragmentinput.linear_centroid, fragmentinput.linear_sample, fragmentinput.perspective, fragmentinput.perspective_centroid, fragmentinput.perspective_sample, fragmentinput.position };
    return fragmentinput_1;
}

void frag_main(FragmentInput_frag_main fragmentinput_frag_main)
{
    FragmentInput val = { fragmentinput_frag_main.position_1, fragmentinput_frag_main._flat_1, fragmentinput_frag_main._linear_1, fragmentinput_frag_main.linear_centroid_1, fragmentinput_frag_main.linear_sample_1, fragmentinput_frag_main.perspective_1, fragmentinput_frag_main.perspective_centroid_1, fragmentinput_frag_main.perspective_sample_1 };
    return;
}
