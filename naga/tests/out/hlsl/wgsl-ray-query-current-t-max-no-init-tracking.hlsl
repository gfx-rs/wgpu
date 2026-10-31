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
    return rq.CommittedRayT();
}

float closest_hit_t(float3 origin, float3 dir, float hit_t)
{
    RayQuery<RAY_FLAG_NONE> rq;
    uint naga_query_init_tracker_for_rq = 0;

    {
        RayDesc_ naga_desc = ConstructRayDesc_(0u, 255u, 0.1, 100.0, origin, dir);
        rq.TraceRayInline(acc_struct, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
    }
    output.Store(0, asuint(GetCurrentTMax(rq, naga_query_init_tracker_for_rq)));
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e13 = false;
        _e13 = rq.Proceed();
        if (_e13) {
        } else {
            break;
        }
        {
            if ((hit_t <= GetCurrentTMax(rq, naga_query_init_tracker_for_rq))) {
                rq.CommitProceduralPrimitiveHit(hit_t);
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
