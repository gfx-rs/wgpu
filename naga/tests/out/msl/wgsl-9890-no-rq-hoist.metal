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

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc [[user(fake0)]]
) {
    uint i = 0u;
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> sq = {};
    uint naga_query_init_tracker_for_sq = 0u;
    float naga_query_tmax_tracker_for_sq = 0.0;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e21 = i;
            i = _e21 + 1u;
        }
        loop_init = false;
        uint _e2 = i;
        if (_e2 < 4u) {
        } else {
            break;
        }
        {
            naga_query_init_tracker_for_sq = 0u;
            uint _e9 = i;
            RayDesc _e18 = RayDesc {1u, 255u, 0.001, 1000.0, metal::float3(static_cast<float>(_e9)), metal::float3(0.0, -1.0, 0.0)};
            {
                RayDesc desc = _e18;
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
                    sq.reset(ray,acc, desc.cull_mask, params);
                    naga_query_init_tracker_for_sq = 1;
                    naga_query_tmax_tracker_for_sq = desc.tmax;
                }
            }
            bool _e19 = false;
            if (((naga_query_init_tracker_for_sq & 1) == 1) && !((naga_query_init_tracker_for_sq & 4) == 4)) {
                _e19 = sq.next();
                naga_query_init_tracker_for_sq = naga_query_init_tracker_for_sq | (_e19 ? 2: 6);
            }
        }
    }
    return;
}
