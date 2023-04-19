struct NagaConstants {
    int base_vertex;
    int base_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct VertexOutput {
    precise float4 position : SV_Position;
    float _varying : LOC1;
};

struct FragmentOutput {
    float depth : SV_Depth;
    uint sample_mask : SV_Coverage;
    float color : SV_Target0;
};

struct Input1_ {
    uint index : SV_VertexID;
};

struct Input2_ {
    uint index : SV_InstanceID;
};

groupshared uint output[1];

struct VertexOutput_vertex {
    float _varying : LOC1;
    precise float4 position : SV_Position;
};

struct FragmentInput_fragment {
    float _varying_1 : LOC1;
    precise float4 position_1 : SV_Position;
    bool front_facing_1 : SV_IsFrontFace;
    uint sample_index_1 : SV_SampleIndex;
    uint sample_mask_1 : SV_Coverage;
};

VertexOutput ConstructVertexOutput(float4 arg0, float arg1) {
    VertexOutput ret = (VertexOutput)0;
    ret.position = arg0;
    ret._varying = arg1;
    return ret;
}

VertexOutput_vertex vertex(uint vertex_index : SV_VertexID, uint instance_index : SV_InstanceID, uint color : LOC10)
{
    uint tmp = (((_NagaConstants.base_vertex + vertex_index) + (_NagaConstants.base_instance + instance_index)) + color);
    const VertexOutput vertexoutput = ConstructVertexOutput((1.0).xxxx, float(tmp));
    const VertexOutput_vertex vertexoutput_1 = { vertexoutput._varying, vertexoutput.position };
    return vertexoutput_1;
}

FragmentOutput ConstructFragmentOutput(float arg0, uint arg1, float arg2) {
    FragmentOutput ret = (FragmentOutput)0;
    ret.depth = arg0;
    ret.sample_mask = arg1;
    ret.color = arg2;
    return ret;
}

FragmentOutput fragment(FragmentInput_fragment fragmentinput_fragment)
{
    VertexOutput in_ = { fragmentinput_fragment.position_1, fragmentinput_fragment._varying_1 };
    bool front_facing = fragmentinput_fragment.front_facing_1;
    uint sample_index = fragmentinput_fragment.sample_index_1;
    uint sample_mask = fragmentinput_fragment.sample_mask_1;
    uint mask = (sample_mask & (1u << sample_index));
    float color_1 = (front_facing ? 1.0 : 0.0);
    const FragmentOutput fragmentoutput = ConstructFragmentOutput(in_._varying, mask, color_1);
    return fragmentoutput;
}

[numthreads(1, 1, 1)]
void compute(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint local_index : SV_GroupIndex, uint3 wg_id : SV_GroupID, uint3 num_wgs : SV_GroupID, uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
        output = (uint[1])0;
    }
    GroupMemoryBarrierWithGroupSync();
    output[0] = ((((global_id.x + local_id.x) + local_index) + wg_id.x) + uint3(_NagaConstants.base_vertex, _NagaConstants.base_instance, _NagaConstants.other).x);
    return;
}

precise float4 vertex_two_structs(Input1_ in1_, Input2_ in2_) : SV_Position
{
    uint index = 2u;

    uint _expr8 = index;
    return float4(float((_NagaConstants.base_vertex + in1_.index)), float((_NagaConstants.base_instance + in2_.index)), float(_expr8), 0.0);
}
