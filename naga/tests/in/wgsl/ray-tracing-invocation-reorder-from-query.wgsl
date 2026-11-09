enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;
enable wgpu_ray_query;

struct Payload {
    color: vec3<f32>,
}

var<ray_payload> payload: Payload;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main() {
    let desc = RayDesc(RAY_FLAG_NONE, 0xffu, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));

    // Traverse with a ray query, then record its committed intersection into a
    // hit object so that the matching closest hit or miss shader can be run.
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, desc);
    while (rayQueryProceed(&rq)) {}

    var ho: hit_object;
    hitObjectRecordFromQuery(&ho, &rq);
    reorderThread(&ho);
    hitObjectExecuteShader(&ho, &payload);
}
