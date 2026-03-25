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
constant float o = 2.0;

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> rq = {};
    RayDesc desc = RayDesc {4u, 255u, 34.0, 38.0, metal::float3(46.0), metal::float3(58.0, 62.0, 74.0)};
    {
        RayDesc desc = desc;
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
        rq.reset(ray,acc_struct, desc.cull_mask, params);
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
