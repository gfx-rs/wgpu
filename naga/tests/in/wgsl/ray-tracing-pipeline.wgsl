enable wgpu_ray_tracing_pipeline;

struct HitCounters {
    hit_num: u32,
    selected_hit: u32,
}

var<ray_payload> hit_num: HitCounters;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main() {
    hit_num = HitCounters();
    traceRay(acc_struct, RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), &hit_num);
}

var<incoming_ray_payload> incoming_hit_num: HitCounters;

@miss
@incoming_payload(incoming_hit_num)
fn miss() {}

@any_hit
@incoming_payload(incoming_hit_num)
fn any_hit_main() {
    incoming_hit_num.hit_num++;
    incoming_hit_num.selected_hit = incoming_hit_num.hit_num;
}

@closest_hit
@incoming_payload(incoming_hit_num)
fn closest_hit_main() {}