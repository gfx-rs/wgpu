struct NoPadding {
    float3 v3_ : LOC0;
    float f3_ : LOC1;
};

struct NeedsPadding {
    float f3_forces_padding : LOC0;
    float3 v3_needs_padding : LOC1;
    float f3_ : LOC2;
};

cbuffer no_padding_uniform : register(b0) { NoPadding no_padding_uniform; }
RWByteAddressBuffer no_padding_storage : register(u1);
cbuffer needs_padding_uniform : register(b2) { NeedsPadding needs_padding_uniform; }
RWByteAddressBuffer needs_padding_storage : register(u3);

struct FragmentInput_no_padding_frag {
    float3 v3_ : LOC0;
    float f3_ : LOC1;
};

struct FragmentInput_needs_padding_frag {
    float f3_forces_padding : LOC0;
    float3 v3_needs_padding : LOC1;
    float f3_1 : LOC2;
};

float4 no_padding_frag(FragmentInput_no_padding_frag fragmentinput_no_padding_frag) : SV_Target0
{
    NoPadding input = { fragmentinput_no_padding_frag.v3_, fragmentinput_no_padding_frag.f3_ };
    return (0.0).xxxx;
}

float4 no_padding_vert(NoPadding input_1) : SV_Position
{
    return (0.0).xxxx;
}

NoPadding ConstructNoPadding(float3 arg0, float arg1) {
    NoPadding ret = (NoPadding)0;
    ret.v3_ = arg0;
    ret.f3_ = arg1;
    return ret;
}

[numthreads(16, 1, 1)]
void no_padding_comp()
{
    NoPadding x = (NoPadding)0;

    NoPadding _e2 = no_padding_uniform;
    x = _e2;
    NoPadding _e4 = ConstructNoPadding(asfloat(no_padding_storage.Load3(0)), asfloat(no_padding_storage.Load(12)));
    x = _e4;
    return;
}

float4 needs_padding_frag(FragmentInput_needs_padding_frag fragmentinput_needs_padding_frag) : SV_Target0
{
    NeedsPadding input_2 = { fragmentinput_needs_padding_frag.f3_forces_padding, fragmentinput_needs_padding_frag.v3_needs_padding, fragmentinput_needs_padding_frag.f3_1 };
    return (0.0).xxxx;
}

float4 needs_padding_vert(NeedsPadding input_3) : SV_Position
{
    return (0.0).xxxx;
}

NeedsPadding ConstructNeedsPadding(float arg0, float3 arg1, float arg2) {
    NeedsPadding ret = (NeedsPadding)0;
    ret.f3_forces_padding = arg0;
    ret.v3_needs_padding = arg1;
    ret.f3_ = arg2;
    return ret;
}

[numthreads(16, 1, 1)]
void needs_padding_comp()
{
    NeedsPadding x_1 = (NeedsPadding)0;

    NeedsPadding _e2 = needs_padding_uniform;
    x_1 = _e2;
    NeedsPadding _e4 = ConstructNeedsPadding(asfloat(needs_padding_storage.Load(0)), asfloat(needs_padding_storage.Load3(16)), asfloat(needs_padding_storage.Load(28)));
    x_1 = _e4;
    return;
}
