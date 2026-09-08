// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct NagaRayQuery {
    thread metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data>& query;
    bool initialized = false;
    bool candidate = false;
    bool finished = false;
    float tmin = 0.0;
    float tmax = 0.0;

    bool next() thread {
        if (!initialized || finished) return false;
        candidate = query.next();
        finished = !candidate;
        return candidate;
    }

    void abort() thread {
        if (candidate) {
            query.abort();
            candidate = false;
            finished = true;
        }
    }

    void commit_triangle_intersection() thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::triangle) {
            query.commit_triangle_intersection();
        }
    }

    void commit_bounding_box_intersection(float distance) thread {
        if (candidate && query.get_candidate_intersection_type() == metal::raytracing::intersection_type::bounding_box) {
            float closest = tmax;
            if (query.get_committed_intersection_type() != metal::raytracing::intersection_type::none) {
                closest = query.get_committed_distance();
            }
            if ((as_type<uint>(distance) & 0x7fffffffu) <= 0x7f800000u && distance >= tmin && distance <= closest) {
                query.commit_bounding_box_intersection(distance);
            }
        }
    }
};

struct RayIntersection {
    uint kind;
    float t;
    uint instance_custom_data;
    uint instance_index;
    uint sbt_record_offset;
    uint geometry_index;
    uint primitive_index;
    metal::float2 barycentrics;
    bool front_face;
    char _pad9[11];
    metal::float4x3 object_to_world;
    metal::float4x3 world_to_object;
};
struct RayDesc {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    metal::float3 origin;
    metal::float3 dir;
};
struct Output {
    uint visible;
    char _pad1[12];
    metal::float3 normal;
};

RayIntersection ray_query_get_intersection_true(thread NagaRayQuery& rq) {
    RayIntersection intersection = RayIntersection {};
    if (!rq.finished) return intersection;
    thread auto& intersector = rq.query;
    metal::raytracing::intersection_type ty = intersector.get_committed_intersection_type();
    if (ty == metal::raytracing::intersection_type::triangle) {
        intersection.kind = 1;
        intersection.barycentrics = intersector.get_committed_triangle_barycentric_coord();
        intersection.front_face = intersector.is_committed_triangle_front_facing();
    } else if (ty == metal::raytracing::intersection_type::bounding_box) {
        intersection.kind = 2;
    }
    if (ty != metal::raytracing::intersection_type::none) {
        intersection.t = intersector.get_committed_distance();
        intersection.instance_custom_data = intersector.get_committed_user_instance_id();
        intersection.instance_index = intersector.get_committed_instance_id();
        intersection.geometry_index = intersector.get_committed_geometry_id();
        intersection.primitive_index = intersector.get_committed_primitive_id();
        intersection.object_to_world = intersector.get_committed_object_to_world_transform();
        intersection.world_to_object = intersector.get_committed_world_to_object_transform();
    }
    return intersection;
}
RayIntersection query_loop(
    metal::float3 pos,
    metal::float3 dir,
    metal::raytracing::instance_acceleration_structure acs
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> naga_ray_query_rq_2;
    NagaRayQuery rq_2 = {naga_ray_query_rq_2};
    RayDesc _e8 = RayDesc {4u, 255u, 0.1, 100.0, pos, dir};
    {
        RayDesc naga_ray_query_desc = _e8;
        thread NagaRayQuery& naga_ray_query_ref = rq_2;
        naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;
        bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);
        bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));
        bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;
        bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & 195u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 304u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 768u) <= 1;
        if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {
            metal::raytracing::intersection_params naga_ray_query_params;
            naga_ray_query_params.set_opacity_cull_mode(
                (naga_ray_query_desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                    (naga_ray_query_desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
                )
            );
            naga_ray_query_params.force_opacity(
                (naga_ray_query_desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (naga_ray_query_desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
            naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & 4) != 0);
            naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);
            naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & 16u) != 0 ? metal::raytracing::triangle_cull_mode::back : ((naga_ray_query_desc.flags & 32u) != 0 ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));
            naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & 256u) != 0 ? metal::raytracing::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & 512u) != 0 ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));
            metal::raytracing::ray naga_ray_query_ray = metal::raytracing::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);
            naga_ray_query_ref.query.reset(naga_ray_query_ray,acs, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    uint2 loop_bound = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e9 = rq_2.next();
        if (_e9) {
        } else {
            break;
        }
    }
    return ray_query_get_intersection_true(rq_2);
}

metal::float3 get_torus_normal(
    metal::float3 world_point,
    RayIntersection intersection
) {
    metal::float3 local_point = intersection.world_to_object * metal::float4(world_point, 1.0);
    metal::float2 point_on_guiding_line = metal::normalize(local_point.xy) * 2.4;
    metal::float3 world_point_on_guiding_line = intersection.object_to_world * metal::float4(point_on_guiding_line, 0.0, 1.0);
    return metal::normalize(world_point - world_point_on_guiding_line);
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
, device Output& output [[user(fake0)]]
) {
    metal::float3 pos_1 = metal::float3(0.0);
    metal::float3 dir_1 = metal::float3(0.0, 1.0, 0.0);
    RayIntersection _e7 = query_loop(pos_1, dir_1, acc_struct);
    output.visible = static_cast<uint>(_e7.kind == 0u);
    metal::float3 _e18 = get_torus_normal(dir_1 * _e7.t, _e7);
    output.normal = _e18;
    return;
}

RayIntersection ray_query_get_intersection_false(thread NagaRayQuery& rq) {
    RayIntersection intersection = RayIntersection {};
    if (!rq.candidate) return intersection;
    thread auto& intersector = rq.query;
    metal::raytracing::intersection_type ty = intersector.get_candidate_intersection_type();
    if (ty == metal::raytracing::intersection_type::triangle) {
        intersection.kind = 1;
        intersection.t = intersector.get_candidate_triangle_distance();
        intersection.barycentrics = intersector.get_candidate_triangle_barycentric_coord();
        intersection.front_face = intersector.is_candidate_triangle_front_facing();
    } else if (ty == metal::raytracing::intersection_type::bounding_box) {
        intersection.kind = 3;
    }
    if (ty != metal::raytracing::intersection_type::none) {
        intersection.instance_custom_data = intersector.get_candidate_user_instance_id();
        intersection.instance_index = intersector.get_candidate_instance_id();
        intersection.geometry_index = intersector.get_candidate_geometry_id();
        intersection.primitive_index = intersector.get_candidate_primitive_id();
        intersection.object_to_world = intersector.get_candidate_object_to_world_transform();
        intersection.world_to_object = intersector.get_candidate_world_to_object_transform();
    }
    return intersection;
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_candidate(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> naga_ray_query_rq;
    NagaRayQuery rq = {naga_ray_query_rq};
    metal::float3 pos_2 = metal::float3(0.0);
    metal::float3 dir_2 = metal::float3(0.0, 1.0, 0.0);
    RayDesc _e12 = RayDesc {4u, 255u, 0.1, 100.0, pos_2, dir_2};
    {
        RayDesc naga_ray_query_desc = _e12;
        thread NagaRayQuery& naga_ray_query_ref = rq;
        naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;
        bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);
        bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));
        bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;
        bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & 195u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 304u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 768u) <= 1;
        if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {
            metal::raytracing::intersection_params naga_ray_query_params;
            naga_ray_query_params.set_opacity_cull_mode(
                (naga_ray_query_desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                    (naga_ray_query_desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
                )
            );
            naga_ray_query_params.force_opacity(
                (naga_ray_query_desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (naga_ray_query_desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
            naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & 4) != 0);
            naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);
            naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & 16u) != 0 ? metal::raytracing::triangle_cull_mode::back : ((naga_ray_query_desc.flags & 32u) != 0 ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));
            naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & 256u) != 0 ? metal::raytracing::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & 512u) != 0 ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));
            metal::raytracing::ray naga_ray_query_ray = metal::raytracing::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);
            naga_ray_query_ref.query.reset(naga_ray_query_ray,acc_struct, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    RayIntersection intersection_1 = ray_query_get_intersection_false(rq);
    if (intersection_1.kind == 3u) {
        rq.commit_bounding_box_intersection(10.0);
        return;
    } else {
        if (intersection_1.kind == 1u) {
            rq.commit_triangle_intersection();
            return;
        } else {
            rq.abort();
            return;
        }
    }
}


[[max_total_threads_per_threadgroup(1)]] kernel void runtime_flags_and_reinitialize(
  metal::raytracing::instance_acceleration_structure acc_struct [[user(fake0)]]
, device Output& output [[user(fake0)]]
) {
    metal::raytracing::intersection_query<metal::raytracing::instancing, metal::raytracing::triangle_data> naga_ray_query_rq_1;
    NagaRayQuery rq_1 = {naga_ray_query_rq_1};
    bool _e1 = rq_1.next();
    rq_1.commit_triangle_intersection();
    rq_1.commit_bounding_box_intersection(1.0);
    rq_1.abort();
    output.visible = ray_query_get_intersection_false(rq_1).kind;
    uint _e10 = output.visible;
    RayDesc _e20 = RayDesc {_e10, 255u, 0.0, 100.0, metal::float3(0.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc naga_ray_query_desc = _e20;
        thread NagaRayQuery& naga_ray_query_ref = rq_1;
        naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;
        bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);
        bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));
        bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;
        bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & 195u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 304u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 768u) <= 1;
        if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {
            metal::raytracing::intersection_params naga_ray_query_params;
            naga_ray_query_params.set_opacity_cull_mode(
                (naga_ray_query_desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                    (naga_ray_query_desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
                )
            );
            naga_ray_query_params.force_opacity(
                (naga_ray_query_desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (naga_ray_query_desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
            naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & 4) != 0);
            naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);
            naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & 16u) != 0 ? metal::raytracing::triangle_cull_mode::back : ((naga_ray_query_desc.flags & 32u) != 0 ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));
            naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & 256u) != 0 ? metal::raytracing::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & 512u) != 0 ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));
            metal::raytracing::ray naga_ray_query_ray = metal::raytracing::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);
            naga_ray_query_ref.query.reset(naga_ray_query_ray,acc_struct, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    uint2 loop_bound_1 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        bool _e21 = rq_1.next();
        if (_e21) {
        } else {
            break;
        }
        {
            RayIntersection hit = ray_query_get_intersection_false(rq_1);
            if (hit.kind == 1u) {
                rq_1.commit_triangle_intersection();
            } else {
                rq_1.commit_bounding_box_intersection(10.0);
            }
        }
    }
    output.visible = ray_query_get_intersection_true(rq_1).kind;
    bool _e31 = rq_1.next();
    RayDesc _e43 = RayDesc {0u, 255u, 1.0, 0.0, metal::float3(0.0), metal::float3(0.0, 0.0, 1.0)};
    {
        RayDesc naga_ray_query_desc = _e43;
        thread NagaRayQuery& naga_ray_query_ref = rq_1;
        naga_ray_query_ref.initialized = false; naga_ray_query_ref.candidate = false; naga_ray_query_ref.finished = false;
        bool naga_ray_query_valid_origin = metal::all((as_type<metal::uint3>(naga_ray_query_desc.origin) & 0x7f800000u) != 0x7f800000u);
        bool naga_ray_query_valid_dir = metal::all((as_type<metal::uint3>(naga_ray_query_desc.dir) & 0x7f800000u) != 0x7f800000u) && metal::any(naga_ray_query_desc.dir != metal::float3(0.0));
        bool naga_ray_query_valid_range = (as_type<uint>(naga_ray_query_desc.tmin) & 0x7f800000u) != 0x7f800000u && (as_type<uint>(naga_ray_query_desc.tmax) & 0x7fffffffu) <= 0x7f800000u && naga_ray_query_desc.tmin >= 0.0 && naga_ray_query_desc.tmax >= naga_ray_query_desc.tmin;
        bool naga_ray_query_valid_flags = metal::popcount(naga_ray_query_desc.flags & 195u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 304u) <= 1 && metal::popcount(naga_ray_query_desc.flags & 768u) <= 1;
        if (naga_ray_query_valid_origin && naga_ray_query_valid_dir && naga_ray_query_valid_range && naga_ray_query_valid_flags) {
            metal::raytracing::intersection_params naga_ray_query_params;
            naga_ray_query_params.set_opacity_cull_mode(
                (naga_ray_query_desc.flags & 64) != 0 ? metal::raytracing::opacity_cull_mode::opaque : (
                    (naga_ray_query_desc.flags & 128) != 0 ? metal::raytracing::opacity_cull_mode::non_opaque : metal::raytracing::opacity_cull_mode::none
                )
            );
            naga_ray_query_params.force_opacity(
                (naga_ray_query_desc.flags & 1) != 0 ? metal::raytracing::forced_opacity::opaque : (
                    (naga_ray_query_desc.flags & 2) != 0 ? metal::raytracing::forced_opacity::non_opaque : metal::raytracing::forced_opacity::none
                )
            );
            naga_ray_query_params.accept_any_intersection((naga_ray_query_desc.flags & 4) != 0);
            naga_ray_query_params.set_triangle_front_facing_winding(metal::winding::clockwise);
            naga_ray_query_params.set_triangle_cull_mode((naga_ray_query_desc.flags & 16u) != 0 ? metal::raytracing::triangle_cull_mode::back : ((naga_ray_query_desc.flags & 32u) != 0 ? metal::raytracing::triangle_cull_mode::front : metal::raytracing::triangle_cull_mode::none));
            naga_ray_query_params.set_geometry_cull_mode((naga_ray_query_desc.flags & 256u) != 0 ? metal::raytracing::geometry_cull_mode::triangle : ((naga_ray_query_desc.flags & 512u) != 0 ? metal::raytracing::geometry_cull_mode::bounding_box : metal::raytracing::geometry_cull_mode::none));
            metal::raytracing::ray naga_ray_query_ray = metal::raytracing::ray(naga_ray_query_desc.origin, naga_ray_query_desc.dir, naga_ray_query_desc.tmin, naga_ray_query_desc.tmax);
            naga_ray_query_ref.query.reset(naga_ray_query_ray,acc_struct, naga_ray_query_desc.cull_mask, naga_ray_query_params);
            naga_ray_query_ref.initialized = true; naga_ray_query_ref.tmin = naga_ray_query_desc.tmin; naga_ray_query_ref.tmax = naga_ray_query_desc.tmax;
        }
    }
    bool _e44 = rq_1.next();
    uint _e47 = output.visible;
    output.visible = _e47 + ray_query_get_intersection_true(rq_1).kind;
    return;
}
