enable wgpu_ray_tracing_pipeline;

struct HitCounters {
    hit_num: u32,
    selected_hit: u32,
}

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    tmin: f32,
    tmax: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

var<ray_payload> hit_num: HitCounters;
@group(0) @binding(0) 
var acc_struct: acceleration_structure;
var<incoming_ray_payload> incoming_hit_num: HitCounters;

@ray_generation 
fn ray_gen_main() {
    hit_num = HitCounters();
    traceRay(acc_struct, RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f)), (&hit_num));
    return;
}

@miss @incoming_payload(incoming_hit_num) 
fn miss() {
    return;
}

@any_hit @incoming_payload(incoming_hit_num) 
fn any_hit_main() {
    let _e3 = incoming_hit_num.hit_num;
    incoming_hit_num.hit_num = (_e3 + 1u);
    let _e9 = incoming_hit_num.hit_num;
    incoming_hit_num.selected_hit = _e9;
    return;
}

@closest_hit @incoming_payload(incoming_hit_num) 
fn closest_hit_main() {
    return;
}
