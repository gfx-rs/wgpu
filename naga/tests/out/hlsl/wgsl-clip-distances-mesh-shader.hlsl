struct VertexOutput {
    float4 position : SV_Position;
    float clip_distances[5] : SV_ClipDistance;
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

groupshared MeshOutput mesh_output;

struct MeshVertexOutput_ms_main {
    float4 position : SV_Position;
    float4 clip_distances : SV_ClipDistance0;
    float clip_distances_1 : SV_ClipDistance1;
};

struct MeshPrimitiveOutput_ms_main {
};

typedef float ret_Constructarray5_float_[5];
ret_Constructarray5_float_ Constructarray5_float_(float arg0, float arg1, float arg2, float arg3, float arg4) {
    float ret[5] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
}

void _ms_main(uint local_invocation_index : SV_GroupIndex)
{
    int i = int(0);

    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices_[0].position = float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices_[1].position = float4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices_[2].position = float4(1.0, -1.0, 0.0, 1.0);
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e50 = i;
            i = asint(asuint(_e50) + asuint(int(1)));
        }
        loop_init = false;
        int _e35 = i;
        if ((_e35 < int(3))) {
        } else {
            break;
        }
        {
            int _e40 = i;
            mesh_output.vertices_[min(uint(_e40), 2u)].clip_distances = Constructarray5_float_(1.0, -1.0, 0.5, -0.5, 0.25);
        }
    }
    mesh_output.primitives_[0].indices_ = uint3(0u, 1u, 2u);
    return;
}
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void ms_main(uint local_invocation_index : SV_GroupIndex, out indices uint3 triangleIndices[1], out vertices MeshVertexOutput_ms_main vertices_[3], out primitives MeshPrimitiveOutput_ms_main primitives_[1]) {
    [loop]
    for (uint zero_flat_index = local_invocation_index; zero_flat_index < 3u; zero_flat_index += 1u) {
        uint zero_index = zero_flat_index;
        mesh_output.vertices_[zero_index].position = (float4)0;
    }
    [loop]
    for (uint zero_flat_index_1 = local_invocation_index; zero_flat_index_1 < 15u; zero_flat_index_1 += 1u) {
        uint zero_index = zero_flat_index_1 / 5u;
        uint zero_index_1 = zero_flat_index_1 % 5u;
        mesh_output.vertices_[zero_index].clip_distances[zero_index_1] = (float)0;
    }
    [loop]
    for (uint zero_flat_index_2 = local_invocation_index; zero_flat_index_2 < 1u; zero_flat_index_2 += 1u) {
        uint zero_index_2 = zero_flat_index_2;
        mesh_output.primitives_[zero_index_2].indices_ = (uint3)0;
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
        vertices_[vertIndex].clip_distances = float4(mesh_output.vertices_[vertIndex].clip_distances[0], mesh_output.vertices_[vertIndex].clip_distances[1], mesh_output.vertices_[vertIndex].clip_distances[2], mesh_output.vertices_[vertIndex].clip_distances[3]);
        vertices_[vertIndex].clip_distances_1 = float(mesh_output.vertices_[vertIndex].clip_distances[4]);
    }
    for (int primIndex = local_invocation_index; primIndex < mesh_output.primitive_count; primIndex += 1) {
        triangleIndices[primIndex] = mesh_output.primitives_[primIndex].indices_;
    }
}

float4 fs_main() : SV_Target0
{
    return (1.0).xxxx;
}
