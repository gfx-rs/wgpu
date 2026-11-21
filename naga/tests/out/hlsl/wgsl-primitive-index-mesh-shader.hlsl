struct VertexOutput {
    float4 position : SV_Position;
};

struct PrimitiveOutput {
    uint3 indices_;
    uint index : SV_PrimitiveID;
};

struct MeshOutput {
    VertexOutput vertices_[3];
    PrimitiveOutput primitives_[1];
    uint vertex_count;
    uint primitive_count;
};

groupshared MeshOutput mesh_output;

struct MeshVertexOutput_ms_main {
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
    uint index_1 : SV_PrimitiveID;
};

struct FragmentInput_fs_main {
    uint index_2 : SV_PrimitiveID;
};

VertexOutput ZeroValueVertexOutput() {
    return (VertexOutput)0;
}

PrimitiveOutput ZeroValuePrimitiveOutput() {
    return (PrimitiveOutput)0;
}

void _ms_main(uint local_invocation_index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    zero_init_index = local_invocation_index;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e50 = zero_init_index;
        if ((_e50 >= 3u)) {
            break;
        }
        mesh_output.vertices_[min(uint(_e50), 2u)] = ZeroValueVertexOutput();
        zero_init_index = (_e50 + 1u);
    }
    mesh_output.primitives_[min(uint(local_invocation_index), 0u)] = ZeroValuePrimitiveOutput();
    if ((local_invocation_index == 0u)) {
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    mesh_output.primitives_[0].index = 7u;
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1]) {
    _ms_main(local_invocation_index);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index; vertIndex < mesh_output.vertex_count; vertIndex += 1) {
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index; primIndex < mesh_output.primitive_count; primIndex += 1) {
        primitives_[primIndex].index_1 = mesh_output.primitives_[primIndex].index;
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    uint index = fragmentinput_fs_main.index_2;
    return float4(float(index), 1.0, 1.0, 1.0);
}
