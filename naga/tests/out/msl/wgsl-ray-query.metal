// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

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
struct RayDesc {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    metal::float3 origin;
    metal::float3 dir;
};
struct Output {
    uint visible;
    char _pad1[12];
    metal::float3 normal;
};

RayIntersection ray_query_get_intersection_true(metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> intersector
, uint intersector_tracker
) {
    RayIntersection intersection = RayIntersection {};
    if (((intersector_tracker & 4) == 4)) {
        metal::raytracing::intersection_type ty = intersector.get_committed_intersection_type();
        if (ty == metal::raytracing::intersection_type::triangle) {
            intersection.kind = 1;
            intersection.barycentrics = intersector.get_committed_triangle_barycentric_coord();
            intersection.front_face = intersector.is_committed_triangle_front_facing();
        } else if (ty == metal::raytracing::intersection_type::bounding_box) {
            intersection.kind = 2;
        }
        if (ty != metal::raytracing::intersection_type::none) {
            intersection.t = intersector.get_committed_distance();
            intersection.instance_custom_data = intersector.get_committed_user_instance_id();
            intersection.instance_index = intersector.get_committed_instance_id();
            intersection.geometry_index = intersector.get_committed_geometry_id();
            intersection.primitive_index = intersector.get_committed_primitive_id();
            intersection.object_to_world = intersector.get_committed_object_to_world_transform();
            intersection.world_to_object = intersector.get_committed_world_to_object_transform();
        }
    }
    return intersection;
}
RayIntersection query_loop(
    metal::float3 pos,
    metal::float3 dir,
    metal::raytracing::instance_acceleration_structure acs
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq_1 = {};
    uint naga_query_init_tracker_for_rq_1 = 0u;
    float naga_query_tmax_tracker_for_rq_1 = 0.0;
    RayDesc _e8 = RayDesc {4u, 255u, 0.1, 100.0, pos, dir};
    {
        RayDesc desc = _e8;
        metal::raytracing::intersection_params params;
        metal::raytracing::opacity_cull_mode cull_mode = 
            (desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                (desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
            );
        params.set_opacity_cull_mode(cull_mode);
        bool force_opacity = cull_mode == metal::raytracing::opacity_cull_mode::none;
        if (force_opacity) {
            params.force_opacity(
                (desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
        }
        params.accept_any_intersection((desc.flags & 4) != 0);
        metal::raytracing::ray ray = metal::raytracing::ray(desc.origin, desc.dir, desc.tmin, desc.tmax);
        bool invalid_nan_infs = ((as_type<uint>(desc.origin.x) & 2139095040) == 2139095040) || ((as_type<uint>(desc.origin.y) & 2139095040) == 2139095040) || ((as_type<uint>(desc.origin.z) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.x) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.y) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.z) & 2139095040) == 2139095040) || ((as_type<uint>(desc.tmin) & 2139095040) == 2139095040) || (((as_type<uint>(desc.tmax) & 2139095040) == 2139095040) && ((as_type<uint>(desc.tmax) & 0x7fffff) != 0));
        bool invalid_t = (desc.tmin > desc.tmax) || (desc.tmin < 0.0);
        bool invalid_dir = metal::all(metal::abs(desc.dir) == 0.0);
        if (!(invalid_dir || invalid_t || invalid_nan_infs)) {
            rq_1.reset(ray,acs, desc.cull_mask, params);
            naga_query_init_tracker_for_rq_1 = 1;
            naga_query_tmax_tracker_for_rq_1 = desc.tmax;
        }
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e9 = false;
        if (((naga_query_init_tracker_for_rq_1 & 1) == 1) && !((naga_query_init_tracker_for_rq_1 & 4) == 4)) {
            _e9 = rq_1.next();
            naga_query_init_tracker_for_rq_1 = naga_query_init_tracker_for_rq_1 | (_e9 ? 2: 6);
        }
        if (_e9) {
        } else {
            break;
        }
    }
    return ray_query_get_intersection_true(rq_1, naga_query_init_tracker_for_rq_1);
}

metal::float3 get_torus_normal(
    metal::float3 world_point,
    RayIntersection intersection
) {
    metal::float3 local_point = intersection.world_to_object * metal::float4(world_point, 1.0);
    metal::float2 point_on_guiding_line = metal::normalize(local_point.xy) * 2.4;
    metal::float3 world_point_on_guiding_line = intersection.object_to_world * metal::float4(point_on_guiding_line, 0.0, 1.0);
    return metal::normalize(world_point - world_point_on_guiding_line);
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
, device Output& output [[user(fake0)]]
) {
    metal::float3 pos_1 = metal::float3(0.0);
    metal::float3 dir_1 = metal::float3(0.0, 1.0, 0.0);
    RayIntersection _e7 = query_loop(pos_1, dir_1, acc_struct);
    output.visible = static_cast<uint>(_e7.kind == 0u);
    metal::float3 _e18 = get_torus_normal(dir_1 * _e7.t, _e7);
    output.normal = _e18;
    return;
}

RayIntersection ray_query_get_intersection_false(metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> intersector
, uint intersector_tracker
) {
    RayIntersection intersection = RayIntersection {};
    if (((intersector_tracker & 2) == 2) && !((intersector_tracker & 4) == 4)) {
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
    }
    return intersection;
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_candidate(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq = {};
    uint naga_query_init_tracker_for_rq = 0u;
    float naga_query_tmax_tracker_for_rq = 0.0;
    metal::float3 pos_2 = metal::float3(0.0);
    metal::float3 dir_2 = metal::float3(0.0, 1.0, 0.0);
    RayDesc _e12 = RayDesc {4u, 255u, 0.1, 100.0, pos_2, dir_2};
    {
        RayDesc desc = _e12;
        metal::raytracing::intersection_params params;
        metal::raytracing::opacity_cull_mode cull_mode = 
            (desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                (desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
            );
        params.set_opacity_cull_mode(cull_mode);
        bool force_opacity = cull_mode == metal::raytracing::opacity_cull_mode::none;
        if (force_opacity) {
            params.force_opacity(
                (desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
        }
        params.accept_any_intersection((desc.flags & 4) != 0);
        metal::raytracing::ray ray = metal::raytracing::ray(desc.origin, desc.dir, desc.tmin, desc.tmax);
        bool invalid_nan_infs = ((as_type<uint>(desc.origin.x) & 2139095040) == 2139095040) || ((as_type<uint>(desc.origin.y) & 2139095040) == 2139095040) || ((as_type<uint>(desc.origin.z) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.x) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.y) & 2139095040) == 2139095040) || ((as_type<uint>(desc.dir.z) & 2139095040) == 2139095040) || ((as_type<uint>(desc.tmin) & 2139095040) == 2139095040) || (((as_type<uint>(desc.tmax) & 2139095040) == 2139095040) && ((as_type<uint>(desc.tmax) & 0x7fffff) != 0));
        bool invalid_t = (desc.tmin > desc.tmax) || (desc.tmin < 0.0);
        bool invalid_dir = metal::all(metal::abs(desc.dir) == 0.0);
        if (!(invalid_dir || invalid_t || invalid_nan_infs)) {
            rq.reset(ray,acc_struct, desc.cull_mask, params);
            naga_query_init_tracker_for_rq = 1;
            naga_query_tmax_tracker_for_rq = desc.tmax;
        }
    }
    RayIntersection intersection_1 = ray_query_get_intersection_false(rq, naga_query_init_tracker_for_rq);
    if (intersection_1.kind == 3u) {
        if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)){
            float t = 10.0;
            float current_max_t = naga_query_tmax_tracker_for_rq;
            if (rq.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
                current_max_t = rq.get_committed_distance();
            }
            if (rq.get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box && (rq.get_ray_min_distance() <= t) && (t <= current_max_t)) {
                rq.commit_bounding_box_intersection(t);
            }
        }
        return;
    } else {
        if (intersection_1.kind == 1u) {
            if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
                if (rq.get_candidate_intersection_type() == metal::raytracing::intersection_type::triangle) {
            rq.commit_triangle_intersection();
                }
            }
            return;
        } else {
            if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
                rq.abort();
            }
            return;
        }
    }
}
