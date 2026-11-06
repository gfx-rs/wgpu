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

void _ms_main(uint local_invocation_index : SV_GroupIndex)
{
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
    [loop]
    for (uint zero_flat_index = local_invocation_index; zero_flat_index < 3u; zero_flat_index += 1u) {
        uint zero_index = (zero_flat_index / 1u) % 3u;
        mesh_output.vertices_[zero_index].position = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_1 = local_invocation_index; zero_flat_index_1 < 1u; zero_flat_index_1 += 1u) {
        uint zero_index_1 = (zero_flat_index_1 / 1u) % 1u;
        mesh_output.primitives_[zero_index_1].indices_ = (uint3)0;
    }
    [loop]
    for (uint zero_flat_index_2 = local_invocation_index; zero_flat_index_2 < 1u; zero_flat_index_2 += 1u) {
        uint zero_index_1 = (zero_flat_index_2 / 1u) % 1u;
        mesh_output.primitives_[zero_index_1].index = (uint)0;
    }
    if (local_invocation_index == 0) {
        mesh_output.vertex_count = (uint)0;
    }
    if (local_invocation_index == 0) {
        mesh_output.primitive_count = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
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
