// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct TaskPayload {
    uint group;
};
struct VertexOutput {
    metal::float4 position;
};
struct PrimitiveOutput {
    metal::uint3 indices;
};
struct TaskInput {
    uint index;
    char _pad1[12];
    metal::uint3 group;
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
struct MeshInput {
    uint index;
    char _pad1[12];
    metal::uint3 group;
};

struct ts_mainInput {
};
metal::uint3 _ts_main(
  uint index_1
, metal::uint3 group
, uint subgroup_size
, object_data TaskPayload& taskPayload
) {
    if (index_1 == 0u) {
        taskPayload = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    const TaskInput input = { index_1, {}, group };
    if (input.index == 0u) {
        taskPayload.group = input.group.x + subgroup_size;
    }
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_main(
  metal::mesh_grid_properties nagaMeshGrid
, uint index_1 [[thread_index_in_threadgroup]]
, metal::uint3 group [[threadgroup_position_in_grid]]
, uint subgroup_size [[threads_per_simdgroup]]
, object_data TaskPayload& taskPayload [[payload]]
) {
    uint3 nagaGridSize = _ts_main(index_1, group, subgroup_size, taskPayload);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (index_1 == 0u) {
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

struct ts_indexInput {
};
metal::uint3 _ts_index(
  uint index
, object_data TaskPayload& taskPayload
) {
    if (index == 0u) {
        taskPayload = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (index == 0u) {
        taskPayload.group = 1u;
    }
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_index(
  metal::mesh_grid_properties nagaMeshGrid_1
, uint index [[thread_index_in_threadgroup]]
, object_data TaskPayload& taskPayload [[payload]]
) {
    uint3 nagaGridSize_1 = _ts_index(index, taskPayload);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (index == 0u) {
        if (
            nagaGridSize_1.x > 256u ||
            nagaGridSize_1.y > 256u ||
            nagaGridSize_1.z > 256u ||
            metal::mulhi(nagaGridSize_1.x, nagaGridSize_1.y) != 0u ||
            metal::mulhi(nagaGridSize_1.x * nagaGridSize_1.y, nagaGridSize_1.z) != 0u ||
            (nagaGridSize_1.x * nagaGridSize_1.y * nagaGridSize_1.z) > 1024u
        ) {
            nagaGridSize_1 = metal::uint3(0u);
        }
        nagaMeshGrid_1.set_threadgroups_per_grid(nagaGridSize_1);
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
  uint index_2
, metal::uint3 group_1
, uint lane
, object_data TaskPayload const& taskPayload
, threadgroup MeshOutput& mesh_output
) {
    if (index_2 == 0u) {
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    const MeshInput input_1 = { index_2, {}, group_1 };
    if (input_1.index == 0u) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    }
    if (input_1.index < 3u) {
        uint _e29 = taskPayload.group;
        mesh_output.vertices.inner[input_1.index].position = metal::float4(static_cast<float>((_e29 + input_1.group.x) + lane), 0.0, 0.0, 1.0);
        return;
    } else {
        return;
    }
}
[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput
, uint index_2 [[thread_index_in_threadgroup]]
, metal::uint3 group_1 [[threadgroup_position_in_grid]]
, uint lane [[thread_index_in_simdgroup]]
, object_data TaskPayload const& taskPayload [[payload]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(index_2, group_1, lane, taskPayload, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = index_2; vertexIndex < metal::min(mesh_output.vertex_count, 3u); vertexIndex += 64) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = index_2; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 64) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 3 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 3 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_index(primitiveIndex * 3 + 2, mesh_output.primitives.inner[primitiveIndex].indices.z);
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (index_2 == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}
