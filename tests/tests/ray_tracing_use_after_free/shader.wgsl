@group(0) @binding(0)
var acc_struct: acceleration_structure;

@workgroup_size(1)
@compute
fn comp_main() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
}