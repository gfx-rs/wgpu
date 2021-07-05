struct VertexOutput {
    float4 position : SV_Position;
    float varying : LOC1;
};

struct FragmentOutput {
    float depth : SV_Depth;
    uint sample_mask : SV_Coverage;
    float color : SV_Target0;
};

struct VertexInput_vertex {
    uint vertex_index1 : SV_VertexID;
    uint instance_index1 : SV_InstanceID;
    uint color1 : LOC10;
};

struct FragmentInput_fragment {
    VertexOutput in2;
    bool front_facing1 : SV_IsFrontFace;
    uint sample_index1 : SV_SampleIndex;
    uint sample_mask1 : SV_Coverage;
};

struct ComputeInput_compute {
    uint3 global_id1 : SV_DispatchThreadID;
    uint3 local_id1 : SV_GroupThreadID;
    uint local_index1 : SV_GroupIndex;
    uint3 wg_id1 : SV_GroupID;
};

VertexOutput vertex(VertexInput_vertex vertexinput_vertex)
{
    uint tmp = ((vertexinput_vertex.vertex_index1 + vertexinput_vertex.instance_index1) + vertexinput_vertex.color1);
    const VertexOutput vertexoutput1 = { float4(1.0.xxxx), float(tmp) };
    return vertexoutput1;
}

FragmentOutput fragment(FragmentInput_fragment fragmentinput_fragment)
{
    uint mask = (fragmentinput_fragment.sample_mask1 & (1u << fragmentinput_fragment.sample_index1));
    float color2 = (fragmentinput_fragment.front_facing1 ? 0.0 : 1.0);
    const FragmentOutput fragmentoutput1 = { fragmentinput_fragment.in2.varying, mask, color2 };
    return fragmentoutput1;
}

[numthreads(1, 1, 1)]
void compute(ComputeInput_compute computeinput_compute)
{
    return;
}
