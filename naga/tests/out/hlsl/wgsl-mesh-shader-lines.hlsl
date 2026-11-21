struct TaskPayload {
    uint dummy;
};

struct VertexOutput {
    float4 position : SV_Position;
};

struct PrimitiveOutput {
    uint2 indices_;
};

struct MeshOutput {
    VertexOutput vertices_[2];
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

TaskPayload ZeroValueTaskPayload() {
    return (TaskPayload)0;
}

uint3 _ts_main(uint local_invocation_index : SV_GroupIndex)
{
    if ((local_invocation_index == 0u)) {
        taskPayload = ZeroValueTaskPayload();
    }
    GroupMemoryBarrierWithGroupSync();
    return uint3(1u, 1u, 1u);
}
[numthreads(64, 1, 1)]
void ts_main(uint local_invocation_index : SV_GroupIndex) {
    uint3 gridSize = _ts_main(local_invocation_index);
    GroupMemoryBarrierWithGroupSync();
    if (
        gridSize.x > 256 ||
        gridSize.y > 256 ||
        gridSize.z > 256 ||
        ((uint64_t)gridSize.x) * ((uint64_t)gridSize.y) > 1024 ||
        ((uint64_t)gridSize.x) * ((uint64_t)gridSize.y) * ((uint64_t)gridSize.z) > 1024
    ) {
        gridSize = uint3(0, 0, 0);
    }
    DispatchMesh(gridSize.x, gridSize.y, gridSize.z, taskPayload);
}

VertexOutput ZeroValueVertexOutput() {
    return (VertexOutput)0;
}

PrimitiveOutput ZeroValuePrimitiveOutput() {
    return (PrimitiveOutput)0;
}

void _ms_main(uint local_invocation_index_1 : SV_GroupIndex, in TaskPayload taskPayload)
{
    if ((local_invocation_index_1 < 2u)) {
        mesh_output.vertices_[min(uint(local_invocation_index_1), 1u)] = ZeroValueVertexOutput();
    }
    if ((local_invocation_index_1 < 1u)) {
        mesh_output.primitives_[min(uint(local_invocation_index_1), 0u)] = ZeroValuePrimitiveOutput();
    }
    if ((local_invocation_index_1 == 0u)) {
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    return;
}
[numthreads(64, 1, 1)]
[outputtopology("line")]
void ms_main(uint local_invocation_index_1 : SV_GroupIndex, out indices uint2 lineIndices[1], out vertices MeshVertexOutput_ms_main vertices_[2], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    _ms_main(local_invocation_index_1, taskPayload);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index_1; vertIndex < mesh_output.vertex_count; vertIndex += 64) {
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index_1; primIndex < mesh_output.primitive_count; primIndex += 64) {
        lineIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}
