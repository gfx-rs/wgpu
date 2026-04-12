enable wgpu_ray_query;

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

struct GpuAabb {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@group(0) @binding(2)
var acc_struct: acceleration_structure;

@group(0) @binding(3)
var<storage, read> gpu_aabbs: array<GpuAabb, 3>;

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let s = sqrt(disc);
    var t = -b - s;
    if (t < 0.001) {
        t = -b + s;
    }
    if (t < 0.001) {
        return -1.0;
    }
    return t;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    var color = vec4<f32>(vec2<f32>(global_id.xy) / vec2<f32>(target_size), 0.0, 1.0);

    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
    let in_uv = pixel_center / vec2<f32>(target_size.xy);
    let d = in_uv * 2.0 - 1.0;

    let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
    let dir = normalize((uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz);

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.01, 200.0, origin, dir));

    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let aabb = gpu_aabbs[c.primitive_index];
            let min_v = vec3<f32>(aabb.min_x, aabb.min_y, aabb.min_z);
            let max_v = vec3<f32>(aabb.max_x, aabb.max_y, aabb.max_z);
            let extent = max_v - min_v;
            let radius = min(extent.x, min(extent.y, extent.z)) * 0.45;
            let center_o = (min_v + max_v) * 0.5;
            let center_w = (c.object_to_world * vec4<f32>(center_o, 1.0)).xyz;
            let t = ray_sphere(origin, dir, center_w, radius);
            if (t > 0.0) {
                rayQueryGenerateIntersection(&rq, t);
            }
        }
    }

    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_GENERATED) {
        let aabb = gpu_aabbs[hit.primitive_index];
        let min_v = vec3<f32>(aabb.min_x, aabb.min_y, aabb.min_z);
        let max_v = vec3<f32>(aabb.max_x, aabb.max_y, aabb.max_z);
        let extent = max_v - min_v;
        let radius = min(extent.x, min(extent.y, extent.z)) * 0.45;
        let center_o = (min_v + max_v) * 0.5;
        let center_w = (hit.object_to_world * vec4<f32>(center_o, 1.0)).xyz;
        let p = origin + dir * hit.t;
        let n = normalize(p - center_w);
        color = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    }

    textureStore(output, global_id.xy, color);
}
