// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct TaskPayload {
    uint dummy;
};
struct VertexOutput {
    metal::float4 position;
};
struct PrimitiveOutput {
    uint indices;
};
struct type_3 {
    VertexOutput inner[1];
};
struct type_4 {
    PrimitiveOutput inner[1];
};
struct MeshOutput {
    type_3 vertices;
    type_4 primitives;
    uint vertex_count;
    uint primitive_count;
    char _pad4[4];
};

struct ts_mainInput {
};
metal::uint3 _ts_main(
  uint local_invocation_index
, object_data TaskPayload& taskPayload
) {
    if (local_invocation_index == 0u) {
        taskPayload = TaskPayload {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_main(
  metal::mesh_grid_properties nagaMeshGrid
, uint local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload& taskPayload [[payload]]
) {
    uint3 nagaGridSize = _ts_main(local_invocation_index, taskPayload);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (local_invocation_index == 0u) {
        if (
            nagaGridSize.x > 256u ||
            nagaGridSize.y > 256u ||
            nagaGridSize.z > 256u ||
            metal::mulhi(nagaGridSize.x, nagaGridSize.y) != 0u ||
            metal::mulhi(nagaGridSize.x * nagaGridSize.y, nagaGridSize.z) != 0u ||
            (nagaGridSize.x * nagaGridSize.y * nagaGridSize.z) > 1024u
        ) {
            nagaGridSize = metal::uint3(0u);
        }
        nagaMeshGrid.set_threadgroups_per_grid(nagaGridSize);
    }
    return;
}

struct ms_mainInput {
};
struct ms_mainVertexOutput {
    metal::float4 position [[position]];
};
struct ms_mainPrimitiveOutput {
};
void _ms_main(
  uint local_invocation_index_1
, object_data TaskPayload const& taskPayload
, threadgroup MeshOutput& mesh_output
) {
    if (local_invocation_index_1 < 1u) {
        mesh_output.vertices.inner[local_invocation_index_1] = VertexOutput {};
        mesh_output.primitives.inner[local_invocation_index_1] = PrimitiveOutput {};
    }
    if (local_invocation_index_1 == 0u) {
        mesh_output.vertex_count = 0u;
        mesh_output.primitive_count = 0u;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    return;
}
[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 1, 1, metal::topology::point> meshOutput
, uint local_invocation_index_1 [[thread_index_in_threadgroup]]
, object_data TaskPayload const& taskPayload [[payload]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(local_invocation_index_1, taskPayload, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = local_invocation_index_1; vertexIndex < metal::min(mesh_output.vertex_count, 1u); vertexIndex += 64) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = local_invocation_index_1; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 64) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 1 + 0, mesh_output.primitives.inner[primitiveIndex].indices);
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (local_invocation_index_1 == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}
