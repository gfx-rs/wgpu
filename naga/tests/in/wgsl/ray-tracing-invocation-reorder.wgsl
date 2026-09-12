enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;

struct Payload {
    color: vec3<f32>,
    hit: u32,
}

struct Hit {
    kind: u32,
    t: f32,
    instance_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: u32,
}

var<ray_payload> payload: Payload;

var<incoming_ray_payload> incoming_payload: Payload;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@group(0) @binding(1)
var<storage, read_write> out_hit: Hit;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>, @builtin(num_ray_invocations) num_invocations: vec3<u32>) {
    var ho: hit_object;

    let shift = vec3<f32>(id) / vec3<f32>(num_invocations);
    let desc = RayDesc(RAY_FLAG_NONE, 0xffu, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0) + shift);

    hitObjectTraceRay(&ho, acc_struct, desc, &payload);

    reorderThread(&ho);
    reorderThread(&ho, 1u, 2u);
    // Abstract integers are automatically converted to `u32`.
    reorderThread(3, 4);

    if hitObjectIsHit(&ho) {
        let intersection = hitObjectGetIntersection(&ho);
        out_hit.kind = intersection.kind;
        out_hit.t = intersection.t;
        out_hit.instance_index = intersection.instance_index;
        out_hit.primitive_index = intersection.primitive_index;
        out_hit.barycentrics = intersection.barycentrics;
        out_hit.front_face = u32(intersection.front_face);
    } else if hitObjectIsMiss(&ho) {
        hitObjectRecordMiss(&ho, desc);
    } else if hitObjectIsEmpty(&ho) {
        hitObjectRecordEmpty(&ho);
    }

    hitObjectExecuteShader(&ho, &payload);
}

@closest_hit
@incoming_payload(incoming_payload)
fn closest_hit_main(@builtin(hit_barycentrics) bary: vec2<f32>) {
    var ho: hit_object;

    incoming_payload.color = vec3(bary, 1.0 - bary.x - bary.y);

    hitObjectTraceRay(
        &ho,
        acc_struct,
        RayDesc(RAY_FLAG_NONE, 0xffu, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)),
        &incoming_payload,
    );
    hitObjectExecuteShader(&ho, &incoming_payload);
}

@miss
@incoming_payload(incoming_payload)
fn miss_main() {
    incoming_payload.hit = 0u;
}
