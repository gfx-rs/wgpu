struct RayDesc_ {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    float3 origin;
    int _pad5_0;
    float3 dir;
    int _end_pad_0;
};

RayDesc RayDescFromRayDesc_(RayDesc_ arg0) {
    RayDesc ret = (RayDesc)0;
    ret.Origin = arg0.origin;
    ret.TMin = arg0.tmin;
    ret.Direction = arg0.dir;
    ret.TMax = arg0.tmax;
    return ret;
}

RaytracingAccelerationStructure acc : register(t0);

RayDesc_ ConstructRayDesc_(uint arg0, uint arg1, float arg2, float arg3, float3 arg4, float3 arg5) {
    RayDesc_ ret = (RayDesc_)0;
    ret.flags = arg0;
    ret.cull_mask = arg1;
    ret.tmin = arg2;
    ret.tmax = arg3;
    ret.origin = arg4;
    ret.dir = arg5;
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    uint i = 0u;
    RayQuery<RAY_FLAG_NONE> sq;
    uint naga_query_init_tracker_for_sq = 0;

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e21 = i;
            i = (_e21 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < 4u)) {
        } else {
            break;
        }
        {
            naga_query_init_tracker_for_sq = 0u;
            uint _e9 = i;
            {
                RayDesc_ naga_desc = ConstructRayDesc_(1u, 255u, 0.001, 1000.0, (float(_e9)).xxx, float3(0.0, -1.0, 0.0));
                float naga_tmin = naga_desc.tmin;
                float naga_tmax = naga_desc.tmax;
                float3 naga_origin = naga_desc.origin;
                float3 naga_dir = naga_desc.dir;
                uint naga_flags = naga_desc.flags;
                bool naga_tmin_valid = (naga_tmin >= 0.0) && (naga_tmin <= naga_tmax) && !(((asuint(naga_tmin) & 2139095040) == 2139095040) && ((asuint(naga_tmin) & 0x7fffff) != 0));
                bool naga_tmax_valid = !(((asuint(naga_tmax) & 2139095040) == 2139095040) && ((asuint(naga_tmax) & 0x7fffff) != 0));
                bool naga_origin_valid = !any((((asuint(naga_origin) & 2139095040) == 2139095040) && ((asuint(naga_origin) & 0x7fffff) != 0)));
                bool naga_dir_valid = !any((((asuint(naga_dir) & 2139095040) == 2139095040) && ((asuint(naga_dir) & 0x7fffff) != 0)));
                bool naga_contains_opaque = ((naga_flags & 1) == 1);
                bool naga_contains_no_opaque = ((naga_flags & 2) == 2);
                bool naga_contains_cull_opaque = ((naga_flags & 64) == 64);
                bool naga_contains_cull_no_opaque = ((naga_flags & 128) == 128);
                bool naga_contains_cull_front = ((naga_flags & 32) == 32);
                bool naga_contains_cull_back = ((naga_flags & 16) == 16);
                bool naga_contains_skip_triangles = ((naga_flags & 256) == 256);
                bool naga_contains_skip_aabbs = ((naga_flags & 512) == 512);
                bool naga_contains_skip_triangles_aabbs =  (naga_contains_skip_aabbs && naga_contains_skip_triangles) ;
                bool naga_contains_skip_triangles_cull =  (naga_contains_cull_front && naga_contains_skip_triangles) || (naga_contains_cull_front && naga_contains_cull_back) || (naga_contains_cull_back && naga_contains_skip_triangles) ;
                bool naga_contains_multiple_opaque =  (naga_contains_cull_no_opaque && naga_contains_opaque) || (naga_contains_cull_no_opaque && naga_contains_no_opaque) || (naga_contains_cull_no_opaque && naga_contains_cull_opaque) || (naga_contains_cull_opaque && naga_contains_opaque) || (naga_contains_cull_opaque && naga_contains_no_opaque) || (naga_contains_no_opaque && naga_contains_opaque) ;
                if (naga_tmin_valid && naga_tmax_valid && naga_origin_valid && naga_dir_valid && !(naga_contains_skip_triangles_aabbs || naga_contains_skip_triangles_cull || naga_contains_multiple_opaque)) {
                    naga_query_init_tracker_for_sq = naga_query_init_tracker_for_sq | 1;
                    sq.TraceRayInline(acc, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
                }
            }
            bool _e19 = false;
            {
                bool naga_has_initialized = ((naga_query_init_tracker_for_sq & 1) == 1);
                bool naga_has_finished = ((naga_query_init_tracker_for_sq & 4) == 4);
                if (naga_has_initialized && !naga_has_finished) {
                    _e19 = sq.Proceed();
                    naga_query_init_tracker_for_sq = naga_query_init_tracker_for_sq | 2;
                    if (!_e19) { naga_query_init_tracker_for_sq = naga_query_init_tracker_for_sq | 4; }
            }}
        }
    }
    return;
}
