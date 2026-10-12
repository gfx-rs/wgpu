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

cbuffer im_vert_block : register(b0) { uint4 im_vert_raw[4]; };
static ImmediateDataVert im_vert;
cbuffer im_frag_block : register(b0) { uint4 im_frag_raw[2]; };
static ImmediateDataFrag im_frag;
ImmediateDataVert ConstructImmediateDataVert(float arg0, float3x3 arg1) {
    ImmediateDataVert ret = (ImmediateDataVert)0;
    ret.position_clip = arg0;
    ret.matrix_ = arg1;
    return ret;
}

ImmediateDataFrag ConstructImmediateDataFrag(float arg0, float4 arg1) {
    ImmediateDataFrag ret = (ImmediateDataFrag)0;
    ret.multiplier = arg0;
    ret.tint = arg1;
    return ret;
}


struct FragmentInput_main {
    float4 color : LOC0;
};

float4 vert_main(float2 pos : LOC0, uint ii : SV_InstanceID, uint vi : SV_VertexID) : SV_Position
{
    im_vert = ConstructImmediateDataVert(asfloat(im_vert_raw[0].x), float3x3(float3(asfloat(im_vert_raw[1].x), asfloat(im_vert_raw[1].y), asfloat(im_vert_raw[1].z)), float3(asfloat(im_vert_raw[2].x), asfloat(im_vert_raw[2].y), asfloat(im_vert_raw[2].z)), float3(asfloat(im_vert_raw[3].x), asfloat(im_vert_raw[3].y), asfloat(im_vert_raw[3].z))));
    float _e9 = im_vert.position_clip;
    return float4(((float((_NagaConstants.first_instance + ii)) * float((_NagaConstants.first_vertex + vi))) * pos), 0.0, _e9);
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn in_ = { fragmentinput_main.color };
    im_frag = ConstructImmediateDataFrag(asfloat(im_frag_raw[0].x), float4(asfloat(im_frag_raw[1].x), asfloat(im_frag_raw[1].y), asfloat(im_frag_raw[1].z), asfloat(im_frag_raw[1].w)));
    float4 _e4 = im_frag.tint;
    return (in_.color * _e4);
}
