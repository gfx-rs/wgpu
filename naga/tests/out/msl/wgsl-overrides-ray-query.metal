// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct NagaRayQuery {
    thread metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data>& query;
    bool initialized = false;
    bool candidate = false;
    bool finished = false;
    float tmin = 0.0;
    float tmax = 0.0;

    bool next() thread {
        if (!initialized || finished) return false;
        candidate = query.next();
        finished = !candidate;
        return candidate;
    }

    void abort() thread {
        if (candidate) {
            query.abort();
            candidate = false;
            finished = true;
        }
    }

    void commit_triangle_intersection() thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::triangle) {
            query.commit_triangle_intersection();
        }
    }

    void commit_bounding_box_intersection(float distance) thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box) {
            float closest = tmax;
            if (query.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
                closest = query.get_committed_distance();
            }
            if ((as_type<uint>(distance) & 0x7fffffffu) <= 0x7f800000u && distance >= tmin && distance <= closest) {
                query.commit_bounding_box_intersection(distance);
            }
        }
    }
};

struct RayDesc {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    metal::float3 origin;
    metal::float3 dir;
};
constant float o = 2.0;

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> naga_ray_query_rq;
    NagaRayQuery rq = {naga_ray_query_rq};
    RayDesc desc = RayDesc {4u, 255u, 34.0, 38.0, metal::float3(46.0), metal::float3(58.0, 62.0, 74.0)};
    {
        RayDesc naga_ray_query_desc = desc;
        thread NagaRayQuery& naga_ray_query_ref = rq;
        naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;
        bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);
        bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));
        bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;
        bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & 195u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 304u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 768u) <= 1;
        if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {
            metal::raytracing::intersection_params naga_ray_query_params;
            naga_ray_query_params.set_opacity_cull_mode(
                (naga_ray_query_desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                    (naga_ray_query_desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
                )
            );
            naga_ray_query_params.force_opacity(
                (naga_ray_query_desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (naga_ray_query_desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
            naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & 4) != 0);
            naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);
            naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & 16u) != 0 ? metal::raytracing::triangle_cull_mode::back : ((naga_ray_query_desc.flags & 32u) != 0 ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));
            naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & 256u) != 0 ? metal::raytracing::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & 512u) != 0 ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));
            metal::raytracing::ray naga_ray_query_ray = metal::raytracing::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);
            naga_ray_query_ref.query.reset(naga_ray_query_ray,acc_struct, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e31 = rq.next();
        if (_e31) {
        } else {
            break;
        }
    }
    return;
}
