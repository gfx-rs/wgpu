// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct _RayQuery {
    metal::raytracing::intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data> intersector;
    metal::raytracing::intersector<metal::raytracing::instancing, metal::raytracing::triangle_data, metal::raytracing::world_space_data>::result_type intersection;
    bool ready = false;
};
constexpr metal::uint _map_intersection_type(const metal::raytracing::intersection_type ty) {
    return ty==metal::raytracing::intersection_type::triangle ? 1 : 
        ty==metal::raytracing::intersection_type::bounding_box ? 4 : 0;
}

struct RayDesc {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    metal::float3 origin;
    metal::float3 dir;
};
struct RayIntersection {
    uint kind;
    float t;
    uint instance_custom_data;
    uint instance_index;
    uint sbt_record_offset;
    uint geometry_index;
    uint primitive_index;
    metal::float2 barycentrics;
    bool front_face;
    char _pad9[11];
    metal::float4x3 object_to_world;
    metal::float4x3 world_to_object;
};

kernel void main_candidate(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    _RayQuery rq_1 = {};
    metal::float3 pos = metal::float3(0.0);
    metal::float3 dir = metal::float3(0.0, 1.0, 0.0);
    RayDesc _e12 = RayDesc {4u, 255u, 0.1, 100.0, pos, dir};
    rq_1.intersector.assume_geometry_type(metal::raytracing::geometry_type::triangle);
    rq_1.intersector.set_opacity_cull_mode((_e12.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (_e12.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none);
    rq_1.intersector.force_opacity((_e12.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (_e12.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none);
    rq_1.intersector.accept_any_intersection((_e12.flags & 4) != 0);
    rq_1.intersection = rq_1.intersector.intersect(metal::raytracing::ray(_e12.origin, _e12.dir, _e12.tmin, _e12.tmax), acc_struct, _e12.cull_mask);    rq_1.ready = true;
    RayIntersection intersection = RayIntersection {_map_intersection_type(rq_1.intersection.type), rq_1.intersection.distance, rq_1.intersection.user_instance_id, rq_1.intersection.instance_id, {}, rq_1.intersection.geometry_id, rq_1.intersection.primitive_id, rq_1.intersection.triangle_barycentric_coord, rq_1.intersection.triangle_front_facing, {}, rq_1.intersection.object_to_world_transform, rq_1.intersection.world_to_object_transform};
    if (intersection.kind == 3u) {
        return;
    } else {
        if (intersection.kind == 1u) {
            return;
        } else {
            rq_1.ready = false;
            return;
        }
    }
}
