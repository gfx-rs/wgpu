struct TaskPayload {
    uint dummy;
};

struct VertexOutput {
    float4 position : SV_Position;
};

struct PrimitiveOutput {
    uint3 indices_;
};

struct MeshOutput {
    VertexOutput vertices_[3];
    PrimitiveOutput primitives_[1];
    uint vertex_count;
    uint primitive_count;
};

groupshared TaskPayload taskPayload;
groupshared MeshOutput mesh_output;

struct MeshVertexOutput_ms_main {
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
};

uint3 _ts_main(uint local_invocation_index : SV_GroupIndex)
{
    return uint3(1u, 1u, 1u);
}
[numthreads(64, 1, 1)]
void ts_main(uint local_invocation_index : SV_GroupIndex) {
    if (local_invocation_index == 0) {
        taskPayload = (TaskPayload)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize = _ts_main(local_invocation_index);
    GroupMemoryBarrierWithGroupSync();
    if (
        gridSize.x > 256 ||
        gridSize.y > 256 ||
        gridSize.z > 256 ||
        ((uint64_t)gridSize.x) * ((uint64_t)gridSize.y) > 0xffffffffull ||
        ((uint64_t)gridSize.x) * ((uint64_t)gridSize.y) * ((uint64_t)gridSize.z) > 1024
    ) {
        gridSize = uint3(0, 0, 0);
    }
    DispatchMesh(gridSize.x, gridSize.y, gridSize.z, taskPayload);
}

void _ms_main(in TaskPayload taskPayload, uint local_invocation_index_1 : SV_GroupIndex)
{
    return;
}
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index_1 : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    if (local_invocation_index_1 == 0) {
        mesh_output = (MeshOutput)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_main(taskPayload, local_invocation_index_1);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index_1; vertIndex < mesh_output.vertex_count; vertIndex += 64) {
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index_1; primIndex < mesh_output.primitive_count; primIndex += 64) {
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}
