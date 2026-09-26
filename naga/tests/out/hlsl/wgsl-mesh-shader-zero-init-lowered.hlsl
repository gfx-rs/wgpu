struct TaskPayload {
    uint visible[64];
    uint count;
};

struct VertexOutput {
    float4 position : SV_Position;
};

struct PrimitiveOutput {
    uint3 indices_;
};

struct MeshOutput {
    VertexOutput vertices_[64];
    PrimitiveOutput primitives_[126];
    uint vertex_count;
    uint primitive_count;
};

groupshared TaskPayload payload_;
groupshared float4 scratch[256];
groupshared MeshOutput mesh_output;

struct MeshVertexOutput_ms_main {
    float4 position : SV_Position;
};

struct MeshPrimitiveOutput_ms_main {
};

uint ZeroValueuint() {
    return (uint)0;
}

float4 ZeroValuefloat4() {
    return (float4)0;
}

uint naga_f2u32(float value) {
    return uint(clamp(value, 0.0, 4294967000.0));
}

uint3 _ts_main(uint index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    zero_init_index = index;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e30 = zero_init_index;
            uint _e31 = (_e30 + 32u);
            zero_init_index = _e31;
            if ((_e31 >= 64u)) {
                break;
            }
        }
        loop_init = false;
        uint _e23 = zero_init_index;
        payload_.visible[min(uint(_e23), 63u)] = ZeroValueuint();
    }
    zero_init_index = index;
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e45 = zero_init_index;
            uint _e46 = (_e45 + 32u);
            zero_init_index = _e46;
            if ((_e46 >= 256u)) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e39 = zero_init_index;
        scratch[min(uint(_e39), 255u)] = ZeroValuefloat4();
    }
    if ((index == 0u)) {
        payload_.count = ZeroValueuint();
    }
    GroupMemoryBarrierWithGroupSync();
    scratch[min(uint(index), 255u)] = (float(index)).xxxx;
    float _e11 = scratch[min(uint(index), 255u)].x;
    payload_.visible[min(uint(index), 63u)] = naga_f2u32(_e11);
    payload_.count = 32u;
    return uint3(1u, 1u, 1u);
}
[numthreads(32, 1, 1)]
void ts_main(uint index : SV_GroupIndex) {
    uint3 gridSize = _ts_main(index);
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
    DispatchMesh(gridSize.x, gridSize.y, gridSize.z, payload_);
}

VertexOutput ZeroValueVertexOutput() {
    return (VertexOutput)0;
}

PrimitiveOutput ZeroValuePrimitiveOutput() {
    return (PrimitiveOutput)0;
}

void _ms_main(uint3 id : SV_GroupThreadID, uint local_invocation_index : SV_GroupIndex, in TaskPayload payload_)
{
    uint zero_init_index_1 = (uint)0;

    zero_init_index_1 = local_invocation_index;
    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e37 = zero_init_index_1;
            uint _e38 = (_e37 + 32u);
            zero_init_index_1 = _e38;
            if ((_e38 >= 64u)) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e30 = zero_init_index_1;
        mesh_output.vertices_[min(uint(_e30), 63u)] = ZeroValueVertexOutput();
    }
    zero_init_index_1 = local_invocation_index;
    uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e49 = zero_init_index_1;
            uint _e50 = (_e49 + 32u);
            zero_init_index_1 = _e50;
            if ((_e50 >= 126u)) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e42 = zero_init_index_1;
        mesh_output.primitives_[min(uint(_e42), 125u)] = ZeroValuePrimitiveOutput();
    }
    if ((local_invocation_index == 0u)) {
        mesh_output.vertex_count = ZeroValueuint();
        mesh_output.primitive_count = ZeroValueuint();
    }
    GroupMemoryBarrierWithGroupSync();
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    uint _e16 = payload_.visible[min(uint(id.x), 63u)];
    mesh_output.vertices_[min(uint(id.x), 63u)].position = (float(_e16)).xxxx;
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    return;
}
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint3 id : SV_GroupThreadID, uint local_invocation_index : SV_GroupIndex, out indices uint3 triangleIndices[126], out vertices MeshVertexOutput_ms_main vertices_[64], out primitives MeshPrimitiveOutput_ms_main primitives_[126], in payload TaskPayload payload_) {
    _ms_main(id, local_invocation_index, payload_);
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(mesh_output.vertex_count, mesh_output.primitive_count);
    for (int vertIndex = local_invocation_index; vertIndex < mesh_output.vertex_count; vertIndex += 32) {
        vertices_[vertIndex].position = mesh_output.vertices_[vertIndex].position;
    }
    for (int primIndex = local_invocation_index; primIndex < mesh_output.primitive_count; primIndex += 32) {
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}
