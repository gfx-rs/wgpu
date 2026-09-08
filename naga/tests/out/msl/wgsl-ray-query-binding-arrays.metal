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

struct _mslBufferSizes {
    uint size3;
};

struct UniformIndex {
    uint index;
};
template <typename T>
struct NagaArgumentBufferWrapper {
    T inner;
};
typedef metal::float2 type_6[1];
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
RayIntersection ray_query_get_intersection_true(thread NagaRayQuery& rq) {
    RayIntersection intersection = RayIntersection {};
    if (!rq.finished) return intersection;
    thread auto& intersector = rq.query;
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
    return intersection;
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  constant NagaArgumentBufferWrapper<metal::raytracing::instance_acceleration_structure>* tlas_array_bounded [[buffer(0)]]
, constant NagaArgumentBufferWrapper<metal::raytracing::instance_acceleration_structure>* tlas_array_unbounded [[buffer(1)]]
, constant UniformIndex& uni [[buffer(2)]]
, device type_6& out [[buffer(3)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> naga_ray_query_rq;
    NagaRayQuery rq = {naga_ray_query_rq};
    uint _e4 = uni.index;
    RayDesc _e18 = RayDesc {4u, 255u, 0.1, 100.0, metal::float3(0.0, 0.0, -2.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc naga_ray_query_desc = _e18;
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
            naga_ray_query_ref.query.reset(naga_ray_query_ray,tlas_array_bounded[_e4].inner, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e19 = rq.next();
        if (_e19) {
        } else {
            break;
        }
    }
    RayIntersection first = ray_query_get_intersection_true(rq);
    uint _e24 = uni.index;
    RayDesc _e37 = RayDesc {4u, 255u, 0.1, 100.0, metal::float3(first.barycentrics, 0.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc naga_ray_query_desc = _e37;
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
            naga_ray_query_ref.query.reset(naga_ray_query_ray,tlas_array_unbounded[_e24].inner, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    uint2 loop_bound_1 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        bool _e38 = rq.next();
        if (_e38) {
        } else {
            break;
        }
    }
    RayIntersection second = ray_query_get_intersection_true(rq);
    uint _e43 = uni.index;
    out[_e43] = metal::float2(first.t, second.t);
    return;
}
