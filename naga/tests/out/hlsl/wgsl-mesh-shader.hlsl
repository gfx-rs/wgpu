struct TaskPayload {
    float4 colorMask;
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
    bool cull : SV_CullPrimitive;
    float4 colorMask : LOC1 : primitive;
};

struct PrimitiveInput {
    float4 colorMask : LOC1 : primitive;
};

struct MeshOutput {
    VertexOutput vertices_[3];
    PrimitiveOutput primitives_[1];
    uint vertex_count;
    uint primitive_count;
};

groupshared TaskPayload taskPayload;
groupshared float workgroupData;
groupshared MeshOutput mesh_output;

struct MeshVertexOutput_ms_main {
    float4 color : LOC0;
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
    float4 colorMask : LOC1 : primitive;
    bool cull : SV_CullPrimitive;
};

struct MeshVertexOutput_ms_no_ts {
    float4 color_1 : LOC0;
    float4 position_1 : SV_Position;
};

struct MeshPrimitiveOutput_ms_no_ts {
    float4 colorMask_1 : LOC1 : primitive;
    bool cull_1 : SV_CullPrimitive;
};

struct MeshVertexOutput_ms_divergent {
    float4 color_2 : LOC0;
    float4 position_2 : SV_Position;
};

struct MeshPrimitiveOutput_ms_divergent {
    float4 colorMask_2 : LOC1 : primitive;
    bool cull_2 : SV_CullPrimitive;
};

struct FragmentInput_fs_main {
    float4 color_3 : LOC0;
    float4 colorMask_3 : LOC1 : primitive;
    float4 position_3 : SV_Position;
};

bool helper_reader(in TaskPayload taskPayload)
{
    bool _e2 = taskPayload.visible;
    return _e2;
}

void helper_writer(bool value)
{
    taskPayload.visible = value;
    return;
}

uint3 _ts_main(uint local_invocation_index : SV_GroupIndex)
{
    workgroupData = 1.0;
    taskPayload.colorMask = float4(1.0, 1.0, 0.0, 1.0);
    helper_writer(true);
    const bool _e12 = helper_reader(taskPayload);
    taskPayload.visible = _e12;
    return uint3(1u, 1u, 1u);
}
[numthreads(1, 1, 1)]
void ts_main(uint local_invocation_index : SV_GroupIndex) {
    if (local_invocation_index == 0) {
        taskPayload = (TaskPayload)0;
        workgroupData = (float)0;
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

uint3 _ts_divergent(uint3 thread_id : SV_GroupThreadID, uint local_invocation_index_1 : SV_GroupIndex)
{
    if ((thread_id.x == 0u)) {
        taskPayload.colorMask = float4(1.0, 1.0, 0.0, 1.0);
        taskPayload.visible = true;
        return uint3(1u, 1u, 1u);
    }
    return uint3(2u, 2u, 2u);
}
[numthreads(2, 1, 1)]
void ts_divergent(uint3 thread_id : SV_GroupThreadID, uint local_invocation_index_1 : SV_GroupIndex) {
    if (local_invocation_index_1 == 0) {
        taskPayload = (TaskPayload)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize_1 = _ts_divergent(thread_id, local_invocation_index_1);
    GroupMemoryBarrierWithGroupSync();
    if (
        gridSize_1.x > 256 ||
        gridSize_1.y > 256 ||
        gridSize_1.z > 256 ||
        ((uint64_t)gridSize_1.x) * ((uint64_t)gridSize_1.y) > 0xffffffffull ||
        ((uint64_t)gridSize_1.x) * ((uint64_t)gridSize_1.y) * ((uint64_t)gridSize_1.z) > 1024
    ) {
        gridSize_1 = uint3(0, 0, 0);
    }
    DispatchMesh(gridSize_1.x, gridSize_1.y, gridSize_1.z, taskPayload);
}

void _ms_main(in TaskPayload taskPayload, uint local_invocation_index_2 : SV_GroupIndex)
{
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2.0;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    float4 _e23 = taskPayload.colorMask;
    mesh_output.vertices_[0].color = (float4(0.0, 1.0, 0.0, 1.0) * _e23);
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    float4 _e45 = taskPayload.colorMask;
    mesh_output.vertices_[1].color = (float4(0.0, 0.0, 1.0, 1.0) * _e45);
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    float4 _e67 = taskPayload.colorMask;
    mesh_output.vertices_[2].color = (float4(1.0, 0.0, 0.0, 1.0) * _e67);
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    const bool _e86 = helper_reader(taskPayload);
    mesh_output.primitives_[0].cull = !(_e86);
    mesh_output.primitives_[0].colorMask = float4(1.0, 0.0, 1.0, 1.0);
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index_2 : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    if (local_invocation_index_2 == 0) {
        workgroupData = (float)0;
        mesh_output = (MeshOutput)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_main(taskPayload, local_invocation_index_2);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index_2; vertIndex < mesh_output.vertex_count; vertIndex += 1) {
        vertices_[vertIndex].color = mesh_output.vertices_[vertIndex].color;
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index_2; primIndex < mesh_output.primitive_count; primIndex += 1) {
        primitives_[primIndex].colorMask = mesh_output.primitives_[primIndex].colorMask;
        primitives_[primIndex].cull = mesh_output.primitives_[primIndex].cull;
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}

void _ms_no_ts(uint local_invocation_index_3 : SV_GroupIndex)
{
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2.0;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices_[0].color = float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices_[1].color = float4(0.0, 0.0, 1.0, 1.0);
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices_[2].color = float4(1.0, 0.0, 0.0, 1.0);
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    mesh_output.primitives_[0].cull = false;
    mesh_output.primitives_[0].colorMask = float4(1.0, 0.0, 1.0, 1.0);
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_no_ts(uint local_invocation_index_3 : SV_GroupIndex, out indices uint3 triangleIndices_1[1], out vertices MeshVertexOutput_ms_no_ts vertices_1[3], out primitives MeshPrimitiveOutput_ms_no_ts primitives_1[1]) {
    if (local_invocation_index_3 == 0) {
        workgroupData = (float)0;
        mesh_output = (MeshOutput)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_no_ts(local_invocation_index_3);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex_1 = local_invocation_index_3; vertIndex_1 < mesh_output.vertex_count; vertIndex_1 += 1) {
        vertices_1[vertIndex_1].color_1 = mesh_output.vertices_[vertIndex_1].color;
        vertices_1[vertIndex_1].position_1 = mesh_output.vertices_[vertIndex_1].position;
    }
    for (int primIndex_1 = local_invocation_index_3; primIndex_1 < mesh_output.primitive_count; primIndex_1 += 1) {
        primitives_1[primIndex_1].colorMask_1 = mesh_output.primitives_[primIndex_1].colorMask;
        primitives_1[primIndex_1].cull_1 = mesh_output.primitives_[primIndex_1].cull;
        triangleIndices_1[primIndex_1] = mesh_output.primitives_[primIndex_1].indices_;
    }
}

void _ms_divergent(uint3 thread_id_1 : SV_GroupThreadID, uint local_invocation_index_4 : SV_GroupIndex)
{
    if ((thread_id_1.x == 0u)) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        workgroupData = 2.0;
        mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
        mesh_output.vertices_[0].color = float4(0.0, 1.0, 0.0, 1.0);
        mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
        mesh_output.vertices_[1].color = float4(0.0, 0.0, 1.0, 1.0);
        mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
        mesh_output.vertices_[2].color = float4(1.0, 0.0, 0.0, 1.0);
        mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
        mesh_output.primitives_[0].cull = false;
        mesh_output.primitives_[0].colorMask = float4(1.0, 0.0, 1.0, 1.0);
        return;
    } else {
        return;
    }
}
[numthreads(2, 1, 1)]
[outputtopology("triangle")]
void ms_divergent(uint3 thread_id_1 : SV_GroupThreadID, uint local_invocation_index_4 : SV_GroupIndex, out indices uint3 triangleIndices_2[1], out vertices MeshVertexOutput_ms_divergent vertices_2[3], out primitives MeshPrimitiveOutput_ms_divergent primitives_2[1]) {
    if (local_invocation_index_4 == 0) {
        workgroupData = (float)0;
        mesh_output = (MeshOutput)0;
    }
    GroupMemoryBarrierWithGroupSync();
    _ms_divergent(thread_id_1, local_invocation_index_4);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex_2 = local_invocation_index_4; vertIndex_2 < mesh_output.vertex_count; vertIndex_2 += 2) {
        vertices_2[vertIndex_2].color_2 = mesh_output.vertices_[vertIndex_2].color;
        vertices_2[vertIndex_2].position_2 = mesh_output.vertices_[vertIndex_2].position;
    }
    for (int primIndex_2 = local_invocation_index_4; primIndex_2 < mesh_output.primitive_count; primIndex_2 += 2) {
        primitives_2[primIndex_2].colorMask_2 = mesh_output.primitives_[primIndex_2].colorMask;
        primitives_2[primIndex_2].cull_2 = mesh_output.primitives_[primIndex_2].cull;
        triangleIndices_2[primIndex_2] = mesh_output.primitives_[primIndex_2].indices_;
    }
}

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    VertexOutput vertex = { fragmentinput_fs_main.position_3, fragmentinput_fs_main.color_3 };
    PrimitiveInput primitive = { fragmentinput_fs_main.colorMask_3 };
    return (vertex.color * primitive.colorMask);
}
