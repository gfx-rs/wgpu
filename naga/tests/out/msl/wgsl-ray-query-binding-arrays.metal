// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

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

struct main_Input {
};
[[max_total_threads_per_threadgroup(2)]] kernel void main_(
  uint local_index [[thread_index_in_threadgroup]]
, constant NagaArgumentBufferWrapper<metal::raytracing::instance_acceleration_structure>* tlas_array_bounded [[buffer(0)]]
, constant NagaArgumentBufferWrapper<metal::raytracing::instance_acceleration_structure>* tlas_array_unbounded [[buffer(1)]]
, constant UniformIndex& uni [[buffer(2)]]
, device type_6& out [[buffer(3)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq = {};
    uint naga_query_init_tracker_for_rq = 0u;
    float naga_query_tmax_tracker_for_rq = 0.0;
    uint _e5 = uni.index;
    RayDesc _e19 = RayDesc {4u, 255u, 0.1, 100.0, metal::float3(0.0, 0.0, -2.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc desc = _e19;
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
            rq.reset(ray,tlas_array_bounded[_e5].inner, desc.cull_mask, params);
            naga_query_init_tracker_for_rq = 1;
            naga_query_tmax_tracker_for_rq = desc.tmax;
        }
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e20 = false;
        if (((naga_query_init_tracker_for_rq & 1) == 1) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
            _e20 = rq.next();
            naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | (_e20 ? 2: 6);
        }
        if (_e20) {
        } else {
            break;
        }
    }
    RayIntersection first = ray_query_get_intersection_true(rq, naga_query_init_tracker_for_rq);
    RayDesc _e35 = RayDesc {4u, 255u, 0.1, 100.0, metal::float3(first.barycentrics, 0.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc desc = _e35;
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
            rq.reset(ray,tlas_array_unbounded[local_index].inner, desc.cull_mask, params);
            naga_query_init_tracker_for_rq = 1;
            naga_query_tmax_tracker_for_rq = desc.tmax;
        }
    }
    uint2 loop_bound_1 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        bool _e36 = false;
        if (((naga_query_init_tracker_for_rq & 1) == 1) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
            _e36 = rq.next();
            naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | (_e36 ? 2: 6);
        }
        if (_e36) {
        } else {
            break;
        }
    }
    RayIntersection second = ray_query_get_intersection_true(rq, naga_query_init_tracker_for_rq);
    out[local_index] = metal::float2(first.t, second.t);
    return;
}
