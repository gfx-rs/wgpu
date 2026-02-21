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
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>, @builtin(num_ray_invocations) num_invocations: vec3<u32>) {
    hit_num = HitCounters();
    let shift = (vec3<f32>(id) / vec3<f32>(num_invocations));
    let ray_shift = ((vec3<f32>(shift.x, 0f, shift.y) * 2f) - vec3(1f));
    traceRay(acc_struct, RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), (vec3<f32>(0f, 1f, 0f) + ray_shift)), (&hit_num));
    return;
}

@miss @incoming_payload(incoming_hit_num) 
fn miss(@builtin(world_ray_origin) origin: vec3<f32>, @builtin(world_ray_direction) dir: vec3<f32>, @builtin(ray_t_min) t_min: f32) {
    return;
}

@any_hit @incoming_payload(incoming_hit_num) 
fn any_hit_main(@builtin(instance_custom_data) data: u32, @builtin(geometry_index) geo_idx: u32, @builtin(ray_t_current_max) max_: f32, @builtin(hit_kind) kind: u32) {
    let _e7 = incoming_hit_num.hit_num;
    incoming_hit_num.hit_num = (_e7 + 1u);
    incoming_hit_num.selected_hit = data;
    return;
}

@closest_hit @incoming_payload(incoming_hit_num) 
fn closest_hit_main(@builtin(object_ray_origin) origin_1: vec3<f32>, @builtin(object_ray_direction) dir_1: vec3<f32>, @builtin(object_to_world) obj_to_world: mat4x3<f32>, @builtin(world_to_object) world_to_obj: mat4x3<f32>) {
    return;
}
