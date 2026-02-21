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

RayIntersection GetCandidateIntersection(RayQuery<RAY_FLAG_NONE> rq, uint rq_tracker) {
    RayIntersection ret = (RayIntersection)0;
    if (((rq_tracker & 2) == 2) && !((rq_tracker & 4) == 4)) {
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
    }
    return ret;
}

[numthreads(1, 1, 1)]
void main_candidate()
{
    RayQuery<RAY_FLAG_NONE> rq_1;
    uint naga_query_init_tracker_for_rq_1 = 0;

    float3 pos = (0.0).xxx;
    float3 dir = float3(0.0, 1.0, 0.0);
    {
        RayDesc_ naga_desc = ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos, dir);
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
            naga_query_init_tracker_for_rq_1 = naga_query_init_tracker_for_rq_1 | 1;
            rq_1.TraceRayInline(acc_struct, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    RayIntersection intersection = GetCandidateIntersection(rq_1, naga_query_init_tracker_for_rq_1);
    if ((intersection.kind == 3u)) {
        if (((naga_query_init_tracker_for_rq_1 & 2) == 2) && !((naga_query_init_tracker_for_rq_1 & 4) == 4)) {
            CANDIDATE_TYPE naga_kind = rq_1.CandidateType();
            float naga_tmin = rq_1.RayTMin();
            float naga_tcurrentmax = rq_1.CommittedRayT();
            if ((naga_kind == CANDIDATE_PROCEDURAL_PRIMITIVE) && (naga_tmin <=10.0) && (10.0 <= naga_tcurrentmax)) {
                rq_1.CommitProceduralPrimitiveHit(10.0);
        }}
        return;
    } else {
        if ((intersection.kind == 1u)) {
            if (((naga_query_init_tracker_for_rq_1 & 2) == 2) && !((naga_query_init_tracker_for_rq_1 & 4) == 4)) {
                CANDIDATE_TYPE naga_kind = rq_1.CandidateType();
                if (naga_kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                    rq_1.CommitNonOpaqueTriangleHit();
            }}
            return;
        } else {
            if (((naga_query_init_tracker_for_rq_1 & 1) == 1)) {
                rq_1.Abort();
            }
            return;
        }
    }
}
