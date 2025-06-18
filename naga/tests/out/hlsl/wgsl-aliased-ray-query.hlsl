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

struct RayIntersection {
    uint kind;
    float t;
    uint instance_custom_data;
    uint instance_index;
    uint sbt_record_offset;
    uint geometry_index;
    uint primitive_index;
    float2 barycentrics;
    bool front_face;
    int _pad9_0;
    int _pad9_1;
    row_major float4x3 object_to_world;
    int _pad10_0;
    row_major float4x3 world_to_object;
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

RayIntersection GetCandidateIntersection(RayQuery<RAY_FLAG_NONE> rq) {
    RayIntersection ret = (RayIntersection)0;
    CANDIDATE_TYPE kind = rq.CandidateType();
    if (kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {
        ret.kind = 1;
        ret.t = rq.CandidateTriangleRayT();
        ret.barycentrics = rq.CandidateTriangleBarycentrics();
        ret.front_face = rq.CandidateTriangleFrontFace();
    } else {
        ret.kind = 3;
    }
    ret.instance_custom_data = rq.CandidateInstanceID();
    ret.instance_index = rq.CandidateInstanceIndex();
    ret.sbt_record_offset = rq.CandidateInstanceContributionToHitGroupIndex();
    ret.geometry_index = rq.CandidateGeometryIndex();
    ret.primitive_index = rq.CandidatePrimitiveIndex();
    ret.object_to_world = rq.CandidateObjectToWorld4x3();
    ret.world_to_object = rq.CandidateWorldToObject4x3();
    return ret;
}

[numthreads(1, 1, 1)]
void main_candidate()
{
    RayQuery<RAY_FLAG_NONE> rq_1;

    float3 pos = (0.0).xxx;
    float3 dir = float3(0.0, 1.0, 0.0);
    rq_1.TraceRayInline(acc_struct, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir).flags, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir).cull_mask, RayDescFromRayDesc_(ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir)));
    RayIntersection intersection = GetCandidateIntersection(rq_1);
    if ((intersection.kind == 3u)) {
        rq_1.CommitProceduralPrimitiveHit(10.0);
        return;
    } else {
        if ((intersection.kind == 1u)) {
            rq_1.CommitNonOpaqueTriangleHit();
            return;
        } else {
            rq_1.Abort();
            return;
        }
    }
}
