struct TaskPayload {
    float4 colors[2];
    bool visible;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : LOC0;
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
    float4 color : LOC0;
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
};

uint3 _ts_main(uint local_invocation_index : SV_GroupIndex)
{
    taskPayload.colors[0] = float4(1.0, 0.0, 0.0, 1.0);
    taskPayload.colors[1] = float4(0.0, 1.0, 0.0, 1.0);
    taskPayload.visible = true;
    return uint3(1u, 1u, 1u);
}
[numthreads(1, 1, 1)]
void ts_main(uint local_invocation_index : SV_GroupIndex) {
    [loop]
    for (uint zero_flat_index = local_invocation_index; zero_flat_index < 2u; zero_flat_index += 1u) {
        uint zero_index = zero_flat_index;
        taskPayload.colors[zero_index] = (float4)0;
    }
    if (local_invocation_index == 0) {
        taskPayload.visible = (bool)0;
    }
    GroupMemoryBarrierWithGroupSync();
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

void _ms_main(in TaskPayload taskPayload, uint local_invocation_index_1 : SV_GroupIndex)
{
    float4 local = (float4)0;
    float4 local_1 = (float4)0;
    float4 local_2 = (float4)0;

    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    float4 _e20 = taskPayload.colors[0];
    local = _e20;
    float4 _e22 = local;
    mesh_output.vertices_[0].color = _e22;
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    float4 _e39 = taskPayload.colors[1];
    local_1 = _e39;
    float4 _e41 = local_1;
    mesh_output.vertices_[1].color = _e41;
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    float4 _e58 = taskPayload.colors[0];
    local_2 = _e58;
    float4 _e60 = local_2;
    mesh_output.vertices_[2].color = _e60;
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index_1 : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    [loop]
    for (uint zero_flat_index_1 = local_invocation_index_1; zero_flat_index_1 < 3u; zero_flat_index_1 += 1u) {
        uint zero_index_1 = zero_flat_index_1;
        mesh_output.vertices_[zero_index_1].position = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_2 = local_invocation_index_1; zero_flat_index_2 < 3u; zero_flat_index_2 += 1u) {
        uint zero_index_1 = zero_flat_index_2;
        mesh_output.vertices_[zero_index_1].color = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_3 = local_invocation_index_1; zero_flat_index_3 < 1u; zero_flat_index_3 += 1u) {
        uint zero_index_2 = zero_flat_index_3;
        mesh_output.primitives_[zero_index_2].indices_ = (uint3)0;
    }
    if (local_invocation_index_1 == 0) {
        mesh_output.vertex_count = (uint)0;
    }
    if (local_invocation_index_1 == 0) {
        mesh_output.primitive_count = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_main(taskPayload, local_invocation_index_1);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index_1; vertIndex < mesh_output.vertex_count; vertIndex += 1) {
        vertices_[vertIndex].color = mesh_output.vertices_[vertIndex].color;
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index_1; primIndex < mesh_output.primitive_count; primIndex += 1) {
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}
