// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct TaskPayload {
    metal::float4 colorMask;
    bool visible;
    char _pad2[15];
};
struct VertexOutput {
    metal::float4 position;
    metal::float4 color;
};
struct PrimitiveOutput {
    metal::packed_uint3 indices;
    bool cull;
    char _pad2[3];
    metal::float4 colorMask;
};
struct PrimitiveInput {
    metal::float4 colorMask;
};
struct type_5 {
    VertexOutput inner[3];
};
struct type_6 {
    PrimitiveOutput inner[1];
};
struct MeshOutput {
    type_5 vertices;
    type_6 primitives;
    uint vertex_count;
    uint primitive_count;
    char _pad4[8];
};

bool helper_reader(
    object_data TaskPayload const& taskPayload
) {
    bool _e2 = taskPayload.visible;
    return _e2;
}

void helper_writer(
    bool value,
    object_data TaskPayload& taskPayload
) {
    taskPayload.visible = value;
    return;
}

metal::uint3 _ts_main(
  uint __local_invocation_index
, object_data TaskPayload& taskPayload
, threadgroup float& workgroupData
) {
    if (__local_invocation_index == 0u) {
        taskPayload = {};
        workgroupData = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    workgroupData = 1.0;
    taskPayload.colorMask = metal::float4(1.0, 1.0, 0.0, 1.0);
    helper_writer(true, taskPayload);
    bool _e12 = helper_reader(taskPayload);
    taskPayload.visible = _e12;
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_main(
  metal::mesh_grid_properties nagaMeshGrid
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload& taskPayload [[payload]]
, threadgroup float& workgroupData
) {
    uint3 nagaGridSize = _ts_main(__local_invocation_index, taskPayload, workgroupData);
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

struct ts_divergentInput {
};
metal::uint3 _ts_divergent(
  metal::uint3 thread_id
, uint __local_invocation_index
, object_data TaskPayload& taskPayload
) {
    if (__local_invocation_index == 0u) {
        taskPayload = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (thread_id.x == 0u) {
        taskPayload.colorMask = metal::float4(1.0, 1.0, 0.0, 1.0);
        taskPayload.visible = true;
        return metal::uint3(1u, 1u, 1u);
    }
    return metal::uint3(2u, 2u, 2u);
}

[[object]] void ts_divergent(
  metal::mesh_grid_properties nagaMeshGrid_1
, metal::uint3 thread_id [[thread_position_in_threadgroup]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload& taskPayload [[payload]]
) {
    uint3 nagaGridSize_1 = _ts_divergent(thread_id, __local_invocation_index, taskPayload);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (__local_invocation_index == 0u) {
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

struct ms_mainVertexOutput {
    metal::float4 position [[position]];
    metal::float4 color [[user(loc0), center_perspective]];
};
struct ms_mainPrimitiveOutput {
    bool cull [[primitive_culled]];
    metal::float4 colorMask [[user(loc1), flat]];
};
void _ms_main(
  uint __local_invocation_index
, object_data TaskPayload const& taskPayload
, threadgroup float& workgroupData
, threadgroup MeshOutput& mesh_output
) {
    if (__local_invocation_index == 0u) {
        workgroupData = {};
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2.0;
    mesh_output.vertices.inner[0].position = metal::float4(0.0, 1.0, 0.0, 1.0);
    metal::float4 _e23 = taskPayload.colorMask;
    mesh_output.vertices.inner[0].color = metal::float4(0.0, 1.0, 0.0, 1.0) * _e23;
    mesh_output.vertices.inner[1].position = metal::float4(-1.0, -1.0, 0.0, 1.0);
    metal::float4 _e45 = taskPayload.colorMask;
    mesh_output.vertices.inner[1].color = metal::float4(0.0, 0.0, 1.0, 1.0) * _e45;
    mesh_output.vertices.inner[2].position = metal::float4(1.0, -1.0, 0.0, 1.0);
    metal::float4 _e67 = taskPayload.colorMask;
    mesh_output.vertices.inner[2].color = metal::float4(1.0, 0.0, 0.0, 1.0) * _e67;
    mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    bool _e86 = helper_reader(taskPayload);
    mesh_output.primitives.inner[0].cull = !(_e86);
    mesh_output.primitives.inner[0].colorMask = metal::float4(1.0, 0.0, 1.0, 1.0);
    return;
}

[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload const& taskPayload [[payload]]
) {
    threadgroup float workgroupData;
    threadgroup MeshOutput mesh_output;
    _ms_main(__local_invocation_index, taskPayload, workgroupData, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = __local_invocation_index; vertexIndex < metal::min(mesh_output.vertex_count, 3u); vertexIndex += 1) {
        ms_mainVertexOutput vertex_1;
        vertex_1.position = mesh_output.vertices.inner[vertexIndex].position;
        vertex_1.color = mesh_output.vertices.inner[vertexIndex].color;
        meshOutput.set_vertex(vertexIndex, vertex_1);
    }
    for(uint primitiveIndex = __local_invocation_index; primitiveIndex < metal::min(mesh_output.primitive_count, 1u); primitiveIndex += 1) {
        ms_mainPrimitiveOutput primitive_1;
        meshOutput.set_index(primitiveIndex * 3 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 3 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_index(primitiveIndex * 3 + 2, mesh_output.primitives.inner[primitiveIndex].indices.z);
        primitive_1.cull = mesh_output.primitives.inner[primitiveIndex].cull;
        primitive_1.colorMask = mesh_output.primitives.inner[primitiveIndex].colorMask;
        meshOutput.set_primitive(primitiveIndex, primitive_1);
    }
    if (__local_invocation_index == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}

struct ms_no_tsVertexOutput {
    metal::float4 position [[position]];
    metal::float4 color [[user(loc0), center_perspective]];
};
struct ms_no_tsPrimitiveOutput {
    bool cull [[primitive_culled]];
    metal::float4 colorMask [[user(loc1), flat]];
};
void _ms_no_ts(
  uint __local_invocation_index
, threadgroup float& workgroupData
, threadgroup MeshOutput& mesh_output
) {
    if (__local_invocation_index == 0u) {
        workgroupData = {};
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2.0;
    mesh_output.vertices.inner[0].position = metal::float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices.inner[0].color = metal::float4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices.inner[1].position = metal::float4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices.inner[1].color = metal::float4(0.0, 0.0, 1.0, 1.0);
    mesh_output.vertices.inner[2].position = metal::float4(1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices.inner[2].color = metal::float4(1.0, 0.0, 0.0, 1.0);
    mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    mesh_output.primitives.inner[0].cull = false;
    mesh_output.primitives.inner[0].colorMask = metal::float4(1.0, 0.0, 1.0, 1.0);
    return;
}

[[mesh]] void ms_no_ts(
  metal::mesh<ms_no_tsVertexOutput, ms_no_tsPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput_1
, uint __local_invocation_index [[thread_index_in_threadgroup]]
) {
    threadgroup float workgroupData;
    threadgroup MeshOutput mesh_output;
    _ms_no_ts(__local_invocation_index, workgroupData, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex_1 = __local_invocation_index; vertexIndex_1 < metal::min(mesh_output.vertex_count, 3u); vertexIndex_1 += 1) {
        ms_no_tsVertexOutput vertex_2;
        vertex_2.position = mesh_output.vertices.inner[vertexIndex_1].position;
        vertex_2.color = mesh_output.vertices.inner[vertexIndex_1].color;
        meshOutput_1.set_vertex(vertexIndex_1, vertex_2);
    }
    for(uint primitiveIndex_1 = __local_invocation_index; primitiveIndex_1 < metal::min(mesh_output.primitive_count, 1u); primitiveIndex_1 += 1) {
        ms_no_tsPrimitiveOutput primitive_2;
        meshOutput_1.set_index(primitiveIndex_1 * 3 + 0, mesh_output.primitives.inner[primitiveIndex_1].indices.x);
        meshOutput_1.set_index(primitiveIndex_1 * 3 + 1, mesh_output.primitives.inner[primitiveIndex_1].indices.y);
        meshOutput_1.set_index(primitiveIndex_1 * 3 + 2, mesh_output.primitives.inner[primitiveIndex_1].indices.z);
        primitive_2.cull = mesh_output.primitives.inner[primitiveIndex_1].cull;
        primitive_2.colorMask = mesh_output.primitives.inner[primitiveIndex_1].colorMask;
        meshOutput_1.set_primitive(primitiveIndex_1, primitive_2);
    }
    if (__local_invocation_index == 0u) {
        meshOutput_1.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}

struct ms_divergentInput {
};
struct ms_divergentVertexOutput {
    metal::float4 position [[position]];
    metal::float4 color [[user(loc0), center_perspective]];
};
struct ms_divergentPrimitiveOutput {
    bool cull [[primitive_culled]];
    metal::float4 colorMask [[user(loc1), flat]];
};
void _ms_divergent(
  metal::uint3 thread_id_1
, uint __local_invocation_index
, threadgroup float& workgroupData
, threadgroup MeshOutput& mesh_output
) {
    if (__local_invocation_index == 0u) {
        workgroupData = {};
        mesh_output = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (thread_id_1.x == 0u) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        workgroupData = 2.0;
        mesh_output.vertices.inner[0].position = metal::float4(0.0, 1.0, 0.0, 1.0);
        mesh_output.vertices.inner[0].color = metal::float4(0.0, 1.0, 0.0, 1.0);
        mesh_output.vertices.inner[1].position = metal::float4(-1.0, -1.0, 0.0, 1.0);
        mesh_output.vertices.inner[1].color = metal::float4(0.0, 0.0, 1.0, 1.0);
        mesh_output.vertices.inner[2].position = metal::float4(1.0, -1.0, 0.0, 1.0);
        mesh_output.vertices.inner[2].color = metal::float4(1.0, 0.0, 0.0, 1.0);
        mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
        mesh_output.primitives.inner[0].cull = false;
        mesh_output.primitives.inner[0].colorMask = metal::float4(1.0, 0.0, 1.0, 1.0);
        return;
    } else {
        return;
    }
}

[[mesh]] void ms_divergent(
  metal::mesh<ms_divergentVertexOutput, ms_divergentPrimitiveOutput, 3, 1, metal::topology::triangle> meshOutput_2
, metal::uint3 thread_id_1 [[thread_position_in_threadgroup]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
) {
    threadgroup float workgroupData;
    threadgroup MeshOutput mesh_output;
    _ms_divergent(thread_id_1, __local_invocation_index, workgroupData, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex_2 = __local_invocation_index; vertexIndex_2 < metal::min(mesh_output.vertex_count, 3u); vertexIndex_2 += 2) {
        ms_divergentVertexOutput vertex_3;
        vertex_3.position = mesh_output.vertices.inner[vertexIndex_2].position;
        vertex_3.color = mesh_output.vertices.inner[vertexIndex_2].color;
        meshOutput_2.set_vertex(vertexIndex_2, vertex_3);
    }
    for(uint primitiveIndex_2 = __local_invocation_index; primitiveIndex_2 < metal::min(mesh_output.primitive_count, 1u); primitiveIndex_2 += 2) {
        ms_divergentPrimitiveOutput primitive_3;
        meshOutput_2.set_index(primitiveIndex_2 * 3 + 0, mesh_output.primitives.inner[primitiveIndex_2].indices.x);
        meshOutput_2.set_index(primitiveIndex_2 * 3 + 1, mesh_output.primitives.inner[primitiveIndex_2].indices.y);
        meshOutput_2.set_index(primitiveIndex_2 * 3 + 2, mesh_output.primitives.inner[primitiveIndex_2].indices.z);
        primitive_3.cull = mesh_output.primitives.inner[primitiveIndex_2].cull;
        primitive_3.colorMask = mesh_output.primitives.inner[primitiveIndex_2].colorMask;
        meshOutput_2.set_primitive(primitiveIndex_2, primitive_3);
    }
    if (__local_invocation_index == 0u) {
        meshOutput_2.set_primitive_count(metal::min(mesh_output.primitive_count, 1u));
    }
}

struct fs_mainInput {
    metal::float4 color [[user(loc0), center_perspective]];
    metal::float4 colorMask [[user(loc1), flat]];
};
struct fs_mainOutput {
    metal::float4 member_5 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings_5 [[stage_in]]
, metal::float4 position_3 [[position]]
) {
    const VertexOutput vertex_ = { position_3, varyings_5.color };
    const PrimitiveInput primitive = { varyings_5.colorMask };
    return fs_mainOutput { vertex_.color * primitive.colorMask };
}
