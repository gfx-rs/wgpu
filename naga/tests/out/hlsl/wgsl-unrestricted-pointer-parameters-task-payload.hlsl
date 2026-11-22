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

float4 ZeroValuefloat4() {
    return (float4)0;
}

uint3 _ts_main(uint local_invocation_index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    zero_init_index = local_invocation_index;
    while(true) {
        uint _e25 = zero_init_index;
        if ((_e25 >= 2u)) {
            break;
        }
        taskPayload.colors[min(uint(_e25), 1u)] = ZeroValuefloat4();
        zero_init_index = (_e25 + 1u);
    }
    if ((local_invocation_index == 0u)) {
        taskPayload.visible = false;
    }
    GroupMemoryBarrierWithGroupSync();
    taskPayload.colors[0] = float4(1.0, 0.0, 0.0, 1.0);
    taskPayload.colors[1] = float4(0.0, 1.0, 0.0, 1.0);
    taskPayload.visible = true;
    return uint3(1u, 1u, 1u);
}
[numthreads(1, 1, 1)]
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
    float4 local = (float4)0;
    float4 local_1 = (float4)0;
    float4 local_2 = (float4)0;
    uint zero_init_index_1 = (uint)0;

    zero_init_index_1 = local_invocation_index_1;
    while(true) {
        uint _e2 = zero_init_index_1;
        if ((_e2 >= 3u)) {
            break;
        }
        mesh_output.vertices_[min(uint(_e2), 2u)] = ZeroValueVertexOutput();
        zero_init_index_1 = (_e2 + 1u);
    }
    mesh_output.primitives_[min(uint(local_invocation_index_1), 0u)] = ZeroValuePrimitiveOutput();
    if ((local_invocation_index_1 == 0u)) {
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    float4 _e39 = taskPayload.colors[0];
    local = _e39;
    float4 _e41 = local;
    mesh_output.vertices_[0].color = _e41;
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    float4 _e58 = taskPayload.colors[1];
    local_1 = _e58;
    float4 _e60 = local_1;
    mesh_output.vertices_[1].color = _e60;
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    float4 _e77 = taskPayload.colors[0];
    local_2 = _e77;
    float4 _e79 = local_2;
    mesh_output.vertices_[2].color = _e79;
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index_1 : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1], in payload TaskPayload taskPayload) {
    _ms_main(local_invocation_index_1, taskPayload);
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
