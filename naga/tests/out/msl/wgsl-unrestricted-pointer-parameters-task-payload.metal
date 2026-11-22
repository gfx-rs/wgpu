// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    metal::float4 inner[2];
};
struct TaskPayload {
    type_1 colors;
    bool visible;
    char _pad2[15];
};
struct VertexOutput {
    metal::float4 position;
    metal::float4 color;
};
struct PrimitiveOutput {
    metal::uint3 indices;
};
struct type_7 {
    VertexOutput inner[3];
};
struct type_8 {
    PrimitiveOutput inner[1];
};
struct MeshOutput {
    type_7 vertices;
    type_8 primitives;
    uint vertex_count;
    uint primitive_count;
    char _pad4[8];
};

void fill(
    object_data TaskPayload& p
) {
    p.colors.inner[0] = metal::float4(1.0, 0.0, 0.0, 1.0);
    p.colors.inner[1] = metal::float4(0.0, 1.0, 0.0, 1.0);
    p.visible = true;
    return;
}

metal::float4 color(
    object_data metal::float4 const& p_1
) {
    metal::float4 _e1 = p_1;
    return _e1;
}

metal::uint3 _ts_main(
  uint __local_invocation_index
, object_data TaskPayload& taskPayload
) {
    if (__local_invocation_index == 0u) {
        taskPayload = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    fill(taskPayload);
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
    metal::float4 color [[user(loc0), center_perspective]];
};
struct ms_mainPrimitiveOutput {
};
void _ms_main(
  uint __local_invocation_index
, object_data TaskPayload const& taskPayload
, threadgroup MeshOutput& mesh_output
) {
    if (__local_invocation_index == 0u) {
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices.inner[0].position = metal::float4(0.0, 1.0, 0.0, 1.0);
    metal::float4 _e22 = color(taskPayload.colors.inner[0]);
    mesh_output.vertices.inner[0].color = _e22;
    mesh_output.vertices.inner[1].position = metal::float4(-1.0, -1.0, 0.0, 1.0);
    metal::float4 _e39 = color(taskPayload.colors.inner[1]);
    mesh_output.vertices.inner[1].color = _e39;
    mesh_output.vertices.inner[2].position = metal::float4(1.0, -1.0, 0.0, 1.0);
    metal::float4 _e56 = color(taskPayload.colors.inner[0]);
    mesh_output.vertices.inner[2].color = _e56;
    mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    return;
}
[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload const& taskPayload [[payload]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(__local_invocation_index, taskPayload, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = __local_invocation_index; vertexIndex < metal::min(mesh_output.vertex_count, 3u); vertexIndex += 1) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        vertex_.color = mesh_output.vertices.inner[vertexIndex].color;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = __local_invocation_index; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 1) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 3 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 3 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_index(primitiveIndex * 3 + 2, mesh_output.primitives.inner[primitiveIndex].indices.z);
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (__local_invocation_index == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}
