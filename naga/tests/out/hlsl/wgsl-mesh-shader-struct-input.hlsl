struct TaskPayload {
    uint group;
};

struct VertexOutput {
    float4 position : SV_Position;
};

struct PrimitiveOutput {
    uint3 indices_;
};

struct TaskInput {
    uint index : SV_GroupIndex;
    uint3 group : SV_GroupID;
};

struct MeshOutput {
    VertexOutput vertices_[3];
    PrimitiveOutput primitives_[1];
    uint vertex_count;
    uint primitive_count;
};

struct MeshInput {
    uint index : SV_GroupIndex;
    uint3 group : SV_GroupID;
};

groupshared TaskPayload taskPayload;
groupshared MeshOutput mesh_output;

struct TaskInput_ts_main {
    uint index_1 : SV_GroupIndex;
    uint3 group : SV_GroupID;
};

struct MeshInput_ms_main {
    uint index_2 : SV_GroupIndex;
    uint3 group_1 : SV_GroupID;
};

struct MeshVertexOutput_ms_main {
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
};

uint3 _ts_main(TaskInput_ts_main taskinput_ts_main)
{
    TaskInput input = { taskinput_ts_main.index_1, taskinput_ts_main.group };
    uint subgroup_size = WaveGetLaneCount();
    if ((input.index == 0u)) {
        taskPayload.group = (input.group.x + subgroup_size);
    }
    return uint3(1u, 1u, 1u);
}
[numthreads(64, 1, 1)]
void ts_main(TaskInput_ts_main taskinput_ts_main) {
    if (taskinput_ts_main.index_1 == 0) {
        taskPayload.group = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize = _ts_main(taskinput_ts_main);
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

uint3 _ts_index(uint index : SV_GroupIndex)
{
    if ((index == 0u)) {
        taskPayload.group = 1u;
    }
    return uint3(1u, 1u, 1u);
}
[numthreads(64, 1, 1)]
void ts_index(uint index : SV_GroupIndex) {
    if (index == 0) {
        taskPayload.group = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize_1 = _ts_index(index);
    GroupMemoryBarrierWithGroupSync();
    if (
        gridSize_1.x > 256 ||
        gridSize_1.y > 256 ||
        gridSize_1.z > 256 ||
        ((uint64_t)gridSize_1.x) * ((uint64_t)gridSize_1.y) > 1024 ||
        ((uint64_t)gridSize_1.x) * ((uint64_t)gridSize_1.y) * ((uint64_t)gridSize_1.z) > 1024
    ) {
        gridSize_1 = uint3(0, 0, 0);
    }
    DispatchMesh(gridSize_1.x, gridSize_1.y, gridSize_1.z, taskPayload);
}

void _ms_main(MeshInput_ms_main meshinput_ms_main, in TaskPayload taskPayload)
{
    MeshInput input_1 = { meshinput_ms_main.index_2, meshinput_ms_main.group_1 };
    uint lane = WaveGetLaneIndex();
    if ((input_1.index == 0u)) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    }
    if ((input_1.index < 3u)) {
        uint _e29 = taskPayload.group;
        mesh_output.vertices_[min(uint(input_1.index), 2u)].position = float4(float(((_e29 + input_1.group.x) + lane)), 0.0, 0.0, 1.0);
        return;
    } else {
        return;
    }
}
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void ms_main(MeshInput_ms_main meshinput_ms_main, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    [loop]
    for (uint zero_flat_index = meshinput_ms_main.index_2; zero_flat_index < 3u; zero_flat_index += 64u) {
        uint zero_index = zero_flat_index;
        mesh_output.vertices_[zero_index].position = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_1 = meshinput_ms_main.index_2; zero_flat_index_1 < 1u; zero_flat_index_1 += 64u) {
        uint zero_index_1 = zero_flat_index_1;
        mesh_output.primitives_[zero_index_1].indices_ = (uint3)0;
    }
    if (meshinput_ms_main.index_2 == 0) {
        mesh_output.vertex_count = (uint)0;
    }
    if (meshinput_ms_main.index_2 == 0) {
        mesh_output.primitive_count = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_main(meshinput_ms_main, taskPayload);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = meshinput_ms_main.index_2; vertIndex < mesh_output.vertex_count; vertIndex += 64) {
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = meshinput_ms_main.index_2; primIndex < mesh_output.primitive_count; primIndex += 64) {
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}
