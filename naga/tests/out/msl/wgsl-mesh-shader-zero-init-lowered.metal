// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    uint inner[64];
};
struct TaskPayload {
    type_1 visible;
    uint count;
};
struct VertexOutput {
    metal::float4 position;
};
struct PrimitiveOutput {
    metal::uint3 indices;
};
struct type_4 {
    VertexOutput inner[64];
};
struct type_5 {
    PrimitiveOutput inner[126];
};
struct MeshOutput {
    type_4 vertices;
    type_5 primitives;
    uint vertex_count;
    uint primitive_count;
    char _pad4[8];
};
struct type_6 {
    metal::float4 inner[256];
};
uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}


struct ts_mainInput {
};
metal::uint3 _ts_main(
  uint index
, object_data TaskPayload& payload
, threadgroup type_6& scratch
) {
    uint zero_init_index = {};
    zero_init_index = index;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e30 = zero_init_index;
            uint _e31 = _e30 + 32u;
            zero_init_index = _e31;
            if (_e31 >= 64u) {
                break;
            }
        }
        loop_init = false;
        uint _e23 = zero_init_index;
        payload.visible.inner[_e23] = uint {};
    }
    zero_init_index = index;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e45 = zero_init_index;
            uint _e46 = _e45 + 32u;
            zero_init_index = _e46;
            if (_e46 >= 256u) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e39 = zero_init_index;
        scratch.inner[_e39] = metal::float4 {};
    }
    if (index == 0u) {
        payload.count = uint {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    scratch.inner[index] = metal::float4(static_cast<float>(index));
    float _e11 = scratch.inner[index].x;
    payload.visible.inner[index] = naga_f2u32(_e11);
    payload.count = 32u;
    return metal::uint3(1u, 1u, 1u);
}

[[object]] void ts_main(
  metal::mesh_grid_properties nagaMeshGrid
, uint index [[thread_index_in_threadgroup]]
, object_data TaskPayload& payload [[payload]]
, threadgroup type_6& scratch
) {
    uint3 nagaGridSize = _ts_main(index, payload, scratch);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    if (index == 0u) {
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
  metal::uint3 id
, uint local_invocation_index
, object_data TaskPayload const& payload
, threadgroup MeshOutput& mesh_output
) {
    uint zero_init_index_1 = {};
    zero_init_index_1 = local_invocation_index;
    uint2 loop_bound_2 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e37 = zero_init_index_1;
            uint _e38 = _e37 + 32u;
            zero_init_index_1 = _e38;
            if (_e38 >= 64u) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e30 = zero_init_index_1;
        mesh_output.vertices.inner[_e30] = VertexOutput {};
    }
    zero_init_index_1 = local_invocation_index;
    uint2 loop_bound_3 = uint2(4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e49 = zero_init_index_1;
            uint _e50 = _e49 + 32u;
            zero_init_index_1 = _e50;
            if (_e50 >= 126u) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e42 = zero_init_index_1;
        mesh_output.primitives.inner[_e42] = PrimitiveOutput {};
    }
    if (local_invocation_index == 0u) {
        mesh_output.vertex_count = uint {};
        mesh_output.primitive_count = uint {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    uint _e16 = payload.visible.inner[id.x];
    mesh_output.vertices.inner[id.x].position = metal::float4(static_cast<float>(_e16));
    mesh_output.primitives.inner[0].indices = metal::uint3(0u, 1u, 2u);
    return;
}
[[mesh]] void ms_main(
  metal::mesh<ms_mainVertexOutput, ms_mainPrimitiveOutput, 64, 126, metal::topology::triangle> meshOutput
, metal::uint3 id [[thread_position_in_threadgroup]]
, uint local_invocation_index [[thread_index_in_threadgroup]]
, object_data TaskPayload const& payload [[payload]]
) {
    threadgroup MeshOutput mesh_output;
    _ms_main(id, local_invocation_index, payload, mesh_output);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::threadgroup_barrier(metal::mem_flags::mem_object_data);
    for(uint vertexIndex = local_invocation_index; vertexIndex < metal::min(mesh_output.vertex_count, 64u); vertexIndex += 32) {
        ms_mainVertexOutput vertex_;
        vertex_.position = mesh_output.vertices.inner[vertexIndex].position;
        meshOutput.set_vertex(vertexIndex, vertex_);
    }
    for(uint primitiveIndex = local_invocation_index; primitiveIndex < metal::min(mesh_output.primitive_count, 126u); primitiveIndex += 32) {
        ms_mainPrimitiveOutput primitive;
        meshOutput.set_index(primitiveIndex * 3 + 0, mesh_output.primitives.inner[primitiveIndex].indices.x);
        meshOutput.set_index(primitiveIndex * 3 + 1, mesh_output.primitives.inner[primitiveIndex].indices.y);
        meshOutput.set_index(primitiveIndex * 3 + 2, mesh_output.primitives.inner[primitiveIndex].indices.z);
        meshOutput.set_primitive(primitiveIndex, primitive);
    }
    if (local_invocation_index == 0u) {
        meshOutput.set_primitive_count(metal::min(mesh_output.primitive_count, 126u));
    }
}
