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

RayIntersection query_loop(float3 pos, float3 dir, RaytracingAccelerationStructure acs)
{
    RayQuery<RAY_FLAG_NONE> rq_1;
    uint naga_query_init_tracker_for_rq_1 = 0;

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
            rq_1.TraceRayInline(acs, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e9 = false;
        {
            bool naga_has_initialized = ((naga_query_init_tracker_for_rq_1 & 1) == 1);
            bool naga_has_finished = ((naga_query_init_tracker_for_rq_1 & 4) == 4);
            if (naga_has_initialized && !naga_has_finished) {
                _e9 = rq_1.Proceed();
                naga_query_init_tracker_for_rq_1 = naga_query_init_tracker_for_rq_1 | 2;
                if (!_e9) { naga_query_init_tracker_for_rq_1 = naga_query_init_tracker_for_rq_1 | 4; }
        }}
        if (_e9) {
        } else {
            break;
        }
        {
        }
    }
    const RayIntersection rayintersection = GetCommittedIntersection(rq_1, naga_query_init_tracker_for_rq_1);
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
    RayQuery<RAY_FLAG_NONE> rq;
    uint naga_query_init_tracker_for_rq = 0;

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
            naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 1;
            rq.TraceRayInline(acc_struct, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    RayIntersection intersection_1 = GetCandidateIntersection(rq, naga_query_init_tracker_for_rq);
    if ((intersection_1.kind == 3u)) {
        if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
            CANDIDATE_TYPE naga_kind = rq.CandidateType();
            float naga_tmin = rq.RayTMin();
            float naga_tcurrentmax = rq.CommittedRayT();
            if ((naga_kind == CANDIDATE_PROCEDURAL_PRIMITIVE) && (naga_tmin <=10.0) && (10.0 <= naga_tcurrentmax)) {
                rq.CommitProceduralPrimitiveHit(10.0);
        }}
        return;
    } else {
        if ((intersection_1.kind == 1u)) {
            if (((naga_query_init_tracker_for_rq & 2) == 2) && !((naga_query_init_tracker_for_rq & 4) == 4)) {
                CANDIDATE_TYPE naga_kind = rq.CandidateType();
                if (naga_kind == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                    rq.CommitNonOpaqueTriangleHit();
            }}
            return;
        } else {
            if (((naga_query_init_tracker_for_rq & 1) == 1)) {
                rq.Abort();
            }
            return;
        }
    }
}
