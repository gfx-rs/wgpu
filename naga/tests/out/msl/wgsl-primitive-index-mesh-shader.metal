// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct VertexOutput {
    metal::float4 position;
};
struct PrimitiveOutput {
    metal::packed_uint3 indices;
    uint index;
};
struct type_3 {
    VertexOutput inner[3];
};
struct type_4 {
    PrimitiveOutput inner[1];
};
struct MeshOutput {
    type_3 vertices;
    type_4 primitives;
    uint vertex_count;
    uint primitive_count;
    char _pad4[8];
};

struct ms_mainVertexOutput {
    metal::float4 position [[position]];
};
struct ms_mainPrimitiveOutput {
    uint index [[primitive_id]];
};
void _ms_main(
  uint __local_invocation_index
, threadgroup MeshOutput& mesh_output
) {
    if (__local_invocation_index == 0u) {
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices.inner[0].position = metal::float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices.inner[1].position = metal::float4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices.inner[2].position = metal::float4(1.0, -1.0, 0.0, 1.0);
    mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    mesh_output.primitives.inner[0].index = 7u;
    return;
}

[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput
, uint __local_invocation_index [[thread_index_in_threadgroup]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(__local_invocation_index, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for(uint vertexIndex = __local_invocation_index; vertexIndex < metal::min(mesh_output.vertex_count, 3u); vertexIndex += 1) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = __local_invocation_index; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 1) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 3 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 3 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_index(primitiveIndex * 3 + 2, mesh_output.primitives.inner[primitiveIndex].indices.z);
        primitive.index = mesh_output.primitives.inner[primitiveIndex].index;
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (__local_invocation_index == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}

struct fs_mainInput {
};
struct fs_mainOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  uint index [[primitive_id]]
) {
    return fs_mainOutput { metal::float4(static_cast<float>(index), 1.0, 1.0, 1.0) };
}
