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
    metal::uint2 indices;
};
struct type_4 {
    VertexOutput inner[2];
};
struct type_5 {
    PrimitiveOutput inner[1];
};
struct MeshOutput {
    type_4 vertices;
    type_5 primitives;
    uint vertex_count;
    uint primitive_count;
};

metal::uint3 _ts_main(
  uint __local_invocation_index
, object_data TaskPayload& taskPayload
) {
    if (__local_invocation_index == 0u) {
        taskPayload = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_main(
  metal::mesh_grid_properties nagaMeshGrid
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload& taskPayload [[payload]]
) {
    uint3 nagaGridSize = _ts_main(__local_invocation_index, taskPayload);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (__local_invocation_index == 0u) {
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

struct ms_mainVertexOutput {
    metal::float4 position [[position]];
};
struct ms_mainPrimitiveOutput {
};
void _ms_main(
  uint __local_invocation_index
, object_data TaskPayload const& taskPayload
, threadgroup MeshOutput* mesh_output
) {
    if (__local_invocation_index == 0u) {
        *mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    return;
}
[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 2, 1, metal::topology::line> meshOutput
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload const& taskPayload [[payload]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(__local_invocation_index, taskPayload, &mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = __local_invocation_index; vertexIndex < metal::min(mesh_output.vertex_count, 2u); vertexIndex += 64) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = __local_invocation_index; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 64) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 2 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 2 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (__local_invocation_index == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}
