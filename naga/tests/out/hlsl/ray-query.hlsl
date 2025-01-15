struct RayIntersection {
    uint kind;
    float t;
    uint instance_custom_index;
    uint instance_id;
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

struct Output {
    uint visible;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    float3 normal;
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

RayIntersection GetCommittedIntersection(RayQuery<RAY_FLAG_NONE> rq) {
    RayIntersection ret = (RayIntersection)0;
    ret.kind = rq.CommittedStatus();
    if( rq.CommittedStatus() == COMMITTED_NOTHING) {} else {
        ret.t = rq.CommittedRayT();
        ret.instance_custom_index = rq.CommittedInstanceID();
        ret.instance_id = rq.CommittedInstanceIndex();
        ret.sbt_record_offset = rq.CommittedInstanceContributionToHitGroupIndex();
        ret.geometry_index = rq.CommittedGeometryIndex();
        ret.primitive_index = rq.CommittedPrimitiveIndex();
        if( rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT ) {
            ret.barycentrics = rq.CommittedTriangleBarycentrics();
            ret.front_face = rq.CommittedTriangleFrontFace();
        }
        ret.object_to_world = rq.CommittedObjectToWorld4x3();
        ret.world_to_object = rq.CommittedWorldToObject4x3();
    }
    return ret;
}

RayIntersection query_loop(float3 pos, float3 dir, RaytracingAccelerationStructure acs)
{
    RayQuery<RAY_FLAG_NONE> rq_1;

    rq_1.TraceRayInline(acs, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir).flags, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir).cull_mask, RayDescFromRayDesc_(ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir)));
    while(true) {
        const bool _e9 = rq_1.Proceed();
        if (_e9) {
        } else {
            break;
        }
        {
        }
    }
    const RayIntersection rayintersection = GetCommittedIntersection(rq_1);
    return rayintersection;
}

float3 get_torus_normal(float3 world_point, RayIntersection intersection)
{
    float3 local_point = mul(float4(world_point, 1.0), intersection.world_to_object);
    float2 point_on_guiding_line = (normalize(local_point.xy) * 2.4);
    float3 world_point_on_guiding_line = mul(float4(point_on_guiding_line, 0.0, 1.0), intersection.object_to_world);
    return normalize((world_point - world_point_on_guiding_line));
}

[numthreads(1, 1, 1)]
void main()
{
    float3 pos_1 = (0.0).xxx;
    float3 dir_1 = float3(0.0, 1.0, 0.0);
    const RayIntersection _e7 = query_loop(pos_1, dir_1, acc_struct);
    output.Store(0, asuint(uint((_e7.kind == 0u))));
    const float3 _e18 = get_torus_normal((dir_1 * _e7.t), _e7);
    output.Store3(16, asuint(_e18));
    return;
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
    ret.instance_custom_index = rq.CandidateInstanceID();
    ret.instance_id = rq.CandidateInstanceIndex();
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
    RayQuery<RAY_FLAG_NONE> rq;

    float3 pos_2 = (0.0).xxx;
    float3 dir_2 = float3(0.0, 1.0, 0.0);
    rq.TraceRayInline(acc_struct, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos_2, dir_2).flags, ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos_2, dir_2).cull_mask, RayDescFromRayDesc_(ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos_2, dir_2)));
    RayIntersection intersection_1 = GetCandidateIntersection(rq);
    output.Store(0, asuint(uint((intersection_1.kind == 3u))));
    return;
}
