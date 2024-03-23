@group(0) @binding(0)
var acc_struct: acceleration_structure;

/*
let RAY_FLAG_NONE = 0x00u;
let RAY_FLAG_FORCE_OPAQUE = 0x01u;
let RAY_FLAG_FORCE_NO_OPAQUE = 0x02u;
let RAY_FLAG_TERMINATE_ON_FIRST_HIT = 0x04u;
let RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 0x08u;
let RAY_FLAG_CULL_BACK_FACING = 0x10u;
let RAY_FLAG_CULL_FRONT_FACING = 0x20u;
let RAY_FLAG_CULL_OPAQUE = 0x40u;
let RAY_FLAG_CULL_NO_OPAQUE = 0x80u;
let RAY_FLAG_SKIP_TRIANGLES = 0x100u;
let RAY_FLAG_SKIP_AABBS = 0x200u;

let RAY_QUERY_INTERSECTION_NONE = 0u;
let RAY_QUERY_INTERSECTION_TRIANGLE = 1u;
let RAY_QUERY_INTERSECTION_GENERATED = 2u;
let RAY_QUERY_INTERSECTION_AABB = 4u;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    t_min: f32,
    t_max: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}
*/

struct Output {
    visible: u32,
    normal: vec3<f32>,
}

@group(0) @binding(1)
var<storage, read_write> output: Output;

fn get_torus_normal(world_point: vec3<f32>, intersection: RayIntersection) -> vec3<f32> {
    let local_point = intersection.world_to_object * vec4<f32>(world_point, 1.0);
    let point_on_guiding_line = normalize(local_point.xy) * 2.4;
    let world_point_on_guiding_line = intersection.object_to_world * vec4<f32>(point_on_guiding_line, 0.0, 1.0);
    return normalize(world_point - world_point_on_guiding_line);
}

@compute @workgroup_size(1)
fn main() {
    var rq: ray_query;

    let dir = vec3<f32>(0.0, 1.0, 0.0);
    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.1, 100.0, vec3<f32>(0.0), dir));

    while (rayQueryProceed(&rq)) {}

    let intersection = rayQueryGetCommittedIntersection(&rq);
    output.visible = u32(intersection.kind == RAY_QUERY_INTERSECTION_NONE);
    output.normal = get_torus_normal(dir * intersection.t, intersection);
}
