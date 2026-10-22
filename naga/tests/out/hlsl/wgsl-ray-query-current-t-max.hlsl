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

RaytracingAccelerationStructure acc_struct : register(t0);
RWByteAddressBuffer output : register(u1);

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

float GetCurrentTMax(RayQuery<RAY_FLAG_NONE> rq, uint rq_tracker) {
    if (!((rq_tracker & 1) == 1)) { return 0.0; }
    return rq.CommittedRayT();
}

float closest_hit_t(float3 origin, float3 dir, float hit_t)
{
    RayQuery<RAY_FLAG_NONE> rq;
    uint naga_query_init_tracker_for_rq = 0;

    {
        RayDesc_ naga_desc = ConstructRayDesc_(0u, 255u, 0.1, 100.0, origin, dir);
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
            naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 1;
            rq.TraceRayInline(acc_struct, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    output.Store(0, asuint(GetCurrentTMax(rq, naga_query_init_tracker_for_rq)));
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e13 = false;
        {
            bool naga_has_initialized = ((naga_query_init_tracker_for_rq & 1) == 1);
            bool naga_has_finished = ((naga_query_init_tracker_for_rq & 4) == 4);
            if (naga_has_initialized && !naga_has_finished) {
                _e13 = rq.Proceed();
                naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 2;
                if (!_e13) { naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 4; }
        }}
        if (_e13) {
        } else {
            break;
        }
        {
            if ((hit_t <= GetCurrentTMax(rq, naga_query_init_tracker_for_rq))) {
                if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
                    CANDIDATE_TYPE naga_kind = rq.CandidateType();
                    float naga_tmin = rq.RayTMin();
                    float naga_tcurrentmax = rq.CommittedRayT();
                    if ((naga_kind == CANDIDATE_PROCEDURAL_PRIMITIVE) && (naga_tmin <=hit_t) && (hit_t <= naga_tcurrentmax)) {
                        rq.CommitProceduralPrimitiveHit(hit_t);
                }}
            }
        }
    }
    return GetCurrentTMax(rq, naga_query_init_tracker_for_rq);
}

[numthreads(1, 1, 1)]
void main()
{
    const float _e9 = closest_hit_t((0.0).xxx, float3(0.0, 1.0, 0.0), 2.0);
    output.Store(4, asuint(_e9));
    return;
}
