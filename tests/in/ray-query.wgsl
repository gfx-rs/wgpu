var acc_struct: acceleration_structure;

struct Output {
    visible: u32,
}
var<storage, read_write> output: Output;

@compute
fn main() {
    var rq: ray_query;

    rayQueryInitialize(rq, acceleration_structure, RAY_FLAGS_TERMINATE_ON_FIRST_HIT, 0xFF, vec3<f32>(0.0), 0.1, vec3<f32>(0.0, 1.0, 0.0), 100.0);

    rayQueryProceed(rq);

    output.visible = rayQueryGetCommittedIntersectionType(rq) == RAY_QUERY_COMMITTED_INTERSECTION_NONE;
}
