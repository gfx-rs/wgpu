// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

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
RayIntersection ray_query_get_intersection_false(metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> intersector) {
    RayIntersection intersection = RayIntersection {};
    metal::raytracing::intersection_type ty = intersector.get_candidate_intersection_type();
    if (ty == metal::raytracing::intersection_type::triangle) {
        intersection.kind = 1;
        intersection.t = intersector.get_candidate_triangle_distance();
        intersection.barycentrics = intersector.get_candidate_triangle_barycentric_coord();
        intersection.front_face = intersector.is_candidate_triangle_front_facing();
    } else if (ty == metal::raytracing::intersection_type::bounding_box) {
        intersection.kind = 3;
    }
    if (ty != metal::raytracing::intersection_type::none) {
        intersection.instance_custom_data = intersector.get_candidate_user_instance_id();
        intersection.instance_index = intersector.get_candidate_instance_id();
        intersection.geometry_index = intersector.get_candidate_geometry_id();
        intersection.primitive_index = intersector.get_candidate_primitive_id();
        intersection.object_to_world = intersector.get_candidate_object_to_world_transform();
        intersection.world_to_object = intersector.get_candidate_world_to_object_transform();
    }
    return intersection;
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_candidate(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq_1 = {};
    metal::float3 pos = metal::float3(0.0);
    metal::float3 dir = metal::float3(0.0, 1.0, 0.0);
    RayDesc _e12 = RayDesc {4u, 255u, 0.1, 100.0, pos, dir};
    {
        RayDesc desc = _e12;
        metal::raytracing::intersection_params params;
        params.set_opacity_cull_mode(
            (desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                (desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
            )
        );
        params.force_opacity(
            (desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                (desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
            )
        );
        params.accept_any_intersection((desc.flags & 4) != 0);
        metal::raytracing::ray ray = metal::raytracing::ray(desc.origin, desc.dir, desc.tmin, desc.tmax);
        rq_1.reset(ray,acc_struct, desc.cull_mask, params);
    }
    RayIntersection intersection = ray_query_get_intersection_false(rq_1);
    if (intersection.kind == 3u) {
        rq_1.commit_bounding_box_intersection(10.0);
        return;
    } else {
        if (intersection.kind == 1u) {
            rq_1.commit_triangle_intersection();
            return;
        } else {
            rq_1.abort();
            return;
        }
    }
}
