@group(0) @binding(0)
var acc_struct: acceleration_structure;

struct Intersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
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
        intersection.instance_custom_index,
        intersection.instance_id,
        intersection.sbt_record_offset,
        intersection.geometry_index,
        intersection.primitive_index,
        intersection.barycentrics,
        u32(intersection.front_face),
        intersection.world_to_object,
        intersection.object_to_world,
    );
}