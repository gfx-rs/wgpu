// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_2 {
    float inner[2];
};
struct RayDesc {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    metal::float3 origin;
    metal::float3 dir;
};

float ray_query_get_current_t_max(metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> intersector, uint intersector_tracker, float t_max) {
    if (!((intersector_tracker & 1) == 1)) {
        return 0.0;
    }
    if (intersector.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
        return intersector.get_committed_distance();
    }
    return t_max;
}

float closest_hit_t(
    metal::float3 origin,
    metal::float3 dir,
    float hit_t,
    metal::raytracing::instance_acceleration_structure acc_struct,
    device type_2& output
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq = {};
    uint naga_query_init_tracker_for_rq = 0u;
    float naga_query_tmax_tracker_for_rq = 0.0;
    RayDesc _e9 = RayDesc {0u, 255u, 0.1, 100.0, origin, dir};
    {
        RayDesc desc = _e9;
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
    output.inner[0] = ray_query_get_current_t_max(rq, naga_query_init_tracker_for_rq, naga_query_tmax_tracker_for_rq);
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e13 = false;
        if (((naga_query_init_tracker_for_rq & 1) == 1) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
            _e13 = rq.next();
            naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | (_e13 ? 2: 6);
        }
        if (_e13) {
        } else {
            break;
        }
        {
            if (hit_t <= ray_query_get_current_t_max(rq, naga_query_init_tracker_for_rq, naga_query_tmax_tracker_for_rq)) {
                if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)){
                    float t = hit_t;
                    float current_max_t = naga_query_tmax_tracker_for_rq;
                    if (rq.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
                        current_max_t = rq.get_committed_distance();
                    }
                    if (rq.get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box && (rq.get_ray_min_distance() <= t) && (t <= current_max_t)) {
                        rq.commit_bounding_box_intersection(t);
                    }
                }
            }
        }
    }
    return ray_query_get_current_t_max(rq, naga_query_init_tracker_for_rq, naga_query_tmax_tracker_for_rq);
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
, device type_2& output [[user(fake0)]]
) {
    float _e9 = closest_hit_t(metal::float3(0.0), metal::float3(0.0, 1.0, 0.0), 2.0, acc_struct, output);
    output.inner[1] = _e9;
    return;
}
