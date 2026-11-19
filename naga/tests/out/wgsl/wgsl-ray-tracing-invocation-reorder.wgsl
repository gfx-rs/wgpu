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

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    tmin: f32,
    tmax: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_data: u32,
    instance_index: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
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
    var i: u32 = 0u;
    var loop_ho: hit_object;

    let shift = (vec3<f32>(id) / vec3<f32>(num_invocations));
    let desc = RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), (vec3<f32>(0f, 1f, 0f) + shift));
    hitObjectTraceRay((&ho), acc_struct, desc, (&payload));
    reorderThread((&ho));
    reorderThread((&ho), 1u, 2u);
    reorderThread(3u, 4u);
    if hitObjectIsHit((&ho)) {
        let intersection = hitObjectGetIntersection((&ho));
        out_hit.kind = intersection.kind;
        out_hit.t = intersection.t;
        out_hit.instance_index = intersection.instance_index;
        out_hit.primitive_index = intersection.primitive_index;
        out_hit.barycentrics = intersection.barycentrics;
        out_hit.front_face = u32(intersection.front_face);
    } else {
        if hitObjectIsMiss((&ho)) {
            hitObjectRecordMiss((&ho), desc);
        } else {
            if hitObjectIsEmpty((&ho)) {
                hitObjectRecordEmpty((&ho));
            }
        }
    }
    loop {
        let _e49 = i;
        if (_e49 < 2u) {
        } else {
            break;
        }
        {
            hitObjectRecordEmpty((&loop_ho));
            if hitObjectIsEmpty((&loop_ho)) {
                hitObjectTraceRay((&loop_ho), acc_struct, desc, (&payload));
            }
            hitObjectExecuteShader((&loop_ho), (&payload));
        }
        continuing {
            let _e57 = i;
            i = (_e57 + 1u);
        }
    }
    hitObjectExecuteShader((&ho), (&payload));
    return;
}

@closest_hit @incoming_payload(incoming_payload)
fn closest_hit_main(@builtin(hit_barycentrics) bary: vec2<f32>) {
    var ho_1: hit_object;

    incoming_payload.color = vec3<f32>(bary, ((1f - bary.x) - bary.y));
    hitObjectTraceRay((&ho_1), acc_struct, RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f)), (&incoming_payload));
    hitObjectExecuteShader((&ho_1), (&incoming_payload));
    return;
}

@miss @incoming_payload(incoming_payload)
fn miss_main() {
    incoming_payload.hit = 0u;
    return;
}
