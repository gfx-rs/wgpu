struct FragmentInput {
    float4 position : SV_Position;
    nointerpolation uint _flat : LOC0;
    nointerpolation uint flat_first : LOC1;
    nointerpolation uint flat_either : LOC2;
    noperspective float _linear : LOC3;
    noperspective centroid float2 linear_centroid : LOC4;
    noperspective sample float3 linear_sample : LOC6;
    noperspective float3 linear_center : LOC7;
    float4 perspective : LOC8;
    centroid float perspective_centroid : LOC9;
    sample float perspective_sample : LOC10;
    float perspective_center : LOC11;
};

struct VertexOutput_vert_main {
    nointerpolation uint _flat_1 : LOC0;
    nointerpolation uint flat_first_1 : LOC1;
    nointerpolation uint flat_either_1 : LOC2;
    noperspective float _linear_1 : LOC3;
    noperspective centroid float2 linear_centroid_1 : LOC4;
    noperspective sample float3 linear_sample_1 : LOC6;
    noperspective float3 linear_center_1 : LOC7;
    float4 perspective_1 : LOC8;
    centroid float perspective_centroid_1 : LOC9;
    sample float perspective_sample_1 : LOC10;
    float perspective_center_1 : LOC11;
    float4 position_1 : SV_Position;
};

struct FragmentInput_frag_main {
    nointerpolation uint _flat_2 : LOC0;
    nointerpolation uint flat_first_2 : LOC1;
    nointerpolation uint flat_either_2 : LOC2;
    noperspective float _linear_2 : LOC3;
    noperspective centroid float2 linear_centroid_2 : LOC4;
    noperspective sample float3 linear_sample_2 : LOC6;
    noperspective float3 linear_center_2 : LOC7;
    float4 perspective_2 : LOC8;
    centroid float perspective_centroid_2 : LOC9;
    sample float perspective_sample_2 : LOC10;
    float perspective_center_2 : LOC11;
    float4 position_2 : SV_Position;
};

struct FragmentInput_frag_main_inline {
    nointerpolation uint _flat_3 : LOC0;
    nointerpolation uint flat_first_3 : LOC1;
    nointerpolation uint flat_either_3 : LOC2;
    noperspective float _linear_3 : LOC3;
    noperspective centroid float2 linear_centroid_3 : LOC4;
    noperspective sample float3 linear_sample_3 : LOC6;
    noperspective float3 linear_center_3 : LOC7;
    float4 perspective_3 : LOC8;
    centroid float perspective_centroid_3 : LOC9;
    sample float perspective_sample_3 : LOC10;
    float perspective_center_3 : LOC11;
    float4 position_3 : SV_Position;
};

VertexOutput_vert_main vert_main()
{
    FragmentInput out_ = (FragmentInput)0;

    out_.position = float4(2.0, 4.0, 5.0, 6.0);
    out_._flat = 8u;
    out_.flat_first = 9u;
    out_.flat_either = 10u;
    out_._linear = 27.0;
    out_.linear_centroid = float2(64.0, 125.0);
    out_.linear_sample = float3(216.0, 343.0, 512.0);
    out_.linear_center = float3(255.0, 511.0, 1024.0);
    out_.perspective = float4(729.0, 1000.0, 1331.0, 1728.0);
    out_.perspective_centroid = 2197.0;
    out_.perspective_sample = 2744.0;
    out_.perspective_center = 2812.0;
    FragmentInput _e41 = out_;
    const FragmentInput fragmentinput = _e41;
    const VertexOutput_vert_main fragmentinput_1 = { fragmentinput._flat, fragmentinput.flat_first, fragmentinput.flat_either, fragmentinput._linear, fragmentinput.linear_centroid, fragmentinput.linear_sample, fragmentinput.linear_center, fragmentinput.perspective, fragmentinput.perspective_centroid, fragmentinput.perspective_sample, fragmentinput.perspective_center, fragmentinput.position };
    return fragmentinput_1;
}

void frag_main(FragmentInput_frag_main fragmentinput_frag_main)
{
    FragmentInput val = { fragmentinput_frag_main.position_2, fragmentinput_frag_main._flat_2, fragmentinput_frag_main.flat_first_2, fragmentinput_frag_main.flat_either_2, fragmentinput_frag_main._linear_2, fragmentinput_frag_main.linear_centroid_2, fragmentinput_frag_main.linear_sample_2, fragmentinput_frag_main.linear_center_2, fragmentinput_frag_main.perspective_2, fragmentinput_frag_main.perspective_centroid_2, fragmentinput_frag_main.perspective_sample_2, fragmentinput_frag_main.perspective_center_2 };
    return;
}

void frag_main_inline(FragmentInput_frag_main_inline fragmentinput_frag_main_inline)
{
    float4 position = fragmentinput_frag_main_inline.position_3;
    uint _flat = fragmentinput_frag_main_inline._flat_3;
    uint flat_first = fragmentinput_frag_main_inline.flat_first_3;
    uint flat_either = fragmentinput_frag_main_inline.flat_either_3;
    float _linear = fragmentinput_frag_main_inline._linear_3;
    float2 linear_centroid = fragmentinput_frag_main_inline.linear_centroid_3;
    float3 linear_sample = fragmentinput_frag_main_inline.linear_sample_3;
    float3 linear_center = fragmentinput_frag_main_inline.linear_center_3;
    float4 perspective = fragmentinput_frag_main_inline.perspective_3;
    float perspective_centroid = fragmentinput_frag_main_inline.perspective_centroid_3;
    float perspective_sample = fragmentinput_frag_main_inline.perspective_sample_3;
    float perspective_center = fragmentinput_frag_main_inline.perspective_center_3;
    return;
}
