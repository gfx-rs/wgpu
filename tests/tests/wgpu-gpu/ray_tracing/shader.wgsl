enable wgpu_ray_query;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

struct Intersection {
    kind: u32,
    t: f32,
    instance_custom_data: u32,
    instance_index: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: u32,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}

@group(0) @binding(1)
var<storage, read_write> out: Intersection;

@workgroup_size(1)
@compute
fn basic_usage() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
}

@workgroup_size(1)
@compute
fn all_of_struct() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.0, 0.0, vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    // this prevents optimisation as we use the fields
    out = Intersection(
        intersection.kind,
        intersection.t,
        intersection.instance_custom_data,
        intersection.instance_index,
        intersection.sbt_record_offset,
        intersection.geometry_index,
        intersection.primitive_index,
        intersection.barycentrics,
        u32(intersection.front_face),
        intersection.world_to_object,
        intersection.object_to_world,
    );
}

struct MaybeInvalidValues {
    nan: f32,
    inf: f32,
}

@group(0) @binding(1)
var<storage> invalid_values: MaybeInvalidValues;

@workgroup_size(1)
@compute
fn invalid_usages() {
    {
        var rq: ray_query;
        // no initialize
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
        // no proceed
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        // The acceleration structure has been set up to not generate an intersections, meaning it will be a committed intersection, not candidate.
        let intersection = rayQueryGetCandidateIntersection(&rq);
    }
    {
        var rq: ray_query;
        // NaN in origin
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, invalid_values.nan, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        // Inf in origin
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, invalid_values.inf, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        // NaN in direction
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, invalid_values.nan, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        // Inf in direction
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, invalid_values.inf, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        // t_min greater than t_max
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 100000.0, 0.1, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        // t_min less than 0
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, -0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
    }
    {
        var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
        rayQueryProceed(&rq);
        // The acceleration structure has been set up to not generate an intersections, meaning terminate is invalid here.
        rayQueryTerminate(&rq);
    }
}