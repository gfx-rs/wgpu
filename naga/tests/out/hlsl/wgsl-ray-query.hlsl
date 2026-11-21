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

RayIntersection GetCommittedIntersection(RayQuery<RAY_FLAG_NONE> rq, uint rq_tracker) {
    RayIntersection ret = (RayIntersection)0;
    if (((rq_tracker & 4) == 4)) {
        ret.kind = rq.CommittedStatus();
        if( rq.CommittedStatus() == COMMITTED_NOTHING) {} else {
            ret.t = rq.CommittedRayT();
            ret.instance_custom_data = rq.CommittedInstanceID();
            ret.instance_index = rq.CommittedInstanceIndex();
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
    }
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    float3 pos = (float3)0;
    float3 dir = (float3)0;
    RayQuery<RAY_FLAG_NONE> rq;
    uint naga_query_init_tracker_for_rq = 0;
    RayIntersection local = (RayIntersection)0;
    float3 world_point = (float3)0;
    RayIntersection intersection = (RayIntersection)0;
    float3 local_1 = (float3)0;

    float3 pos_1 = (0.0).xxx;
    float3 dir_1 = float3(0.0, 1.0, 0.0);
    pos = pos_1;
    float3 _e7 = pos;
    dir = dir_1;
    float3 _e9 = dir;
    {
        RayDesc_ naga_desc = ConstructRayDesc_(4u, 255u, 0.1, 100.0, _e7, _e9);
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
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e51 = false;
        {
            bool naga_has_initialized = ((naga_query_init_tracker_for_rq & 1) == 1);
            bool naga_has_finished = ((naga_query_init_tracker_for_rq & 4) == 4);
            if (naga_has_initialized && !naga_has_finished) {
                _e51 = rq.Proceed();
                naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 2;
                if (!_e51) { naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 4; }
        }}
        if (_e51) {
        } else {
            break;
        }
        {
        }
    }
    local = GetCommittedIntersection(rq, naga_query_init_tracker_for_rq);
    RayIntersection intersection_1 = local;
    output.Store(0, asuint(uint((intersection_1.kind == 0u))));
    world_point = (dir_1 * intersection_1.t);
    float3 _e30 = world_point;
    intersection = intersection_1;
    RayIntersection _e32 = intersection;
    local_1 = normalize((_e30 - mul(float4((normalize(mul(float4(_e30, 1.0), _e32.world_to_object).xy) * 2.4), 0.0, 1.0), _e32.object_to_world)));
    float3 _e49 = local_1;
    output.Store3(16, asuint(_e49));
    return;
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

    float3 pos_2 = (0.0).xxx;
    float3 dir_2 = float3(0.0, 1.0, 0.0);
    {
        RayDesc_ naga_desc = ConstructRayDesc_(4u, 255u, 0.1, 100.0, pos_2, dir_2);
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
    RayIntersection intersection_2 = GetCandidateIntersection(rq_1, naga_query_init_tracker_for_rq_1);
    if ((intersection_2.kind == 3u)) {
        if (((naga_query_init_tracker_for_rq_1 & 2) == 2) && !((naga_query_init_tracker_for_rq_1 & 4) == 4)) {
            CANDIDATE_TYPE naga_kind = rq_1.CandidateType();
            float naga_tmin = rq_1.RayTMin();
            float naga_tcurrentmax = rq_1.CommittedRayT();
            if ((naga_kind == CANDIDATE_PROCEDURAL_PRIMITIVE) && (naga_tmin <=10.0) && (10.0 <= naga_tcurrentmax)) {
                rq_1.CommitProceduralPrimitiveHit(10.0);
        }}
        return;
    } else {
        if ((intersection_2.kind == 1u)) {
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
