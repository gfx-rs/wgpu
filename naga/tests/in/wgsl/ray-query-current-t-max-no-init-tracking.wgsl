enable wgpu_ray_query;

/// The scene to trace against.
@group(0) @binding(0)
var acc_struct: acceleration_structure;

/// Receives the values that the shader reads.
@group(0) @binding(1)
var<storage, read_write> output: array<f32, 2>;

/// Generates a hit at `hit_t` when it is in range, and returns the final current t max.
fn closest_hit_t(origin: vec3<f32>, dir: vec3<f32>, hit_t: f32) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(RAY_FLAG_NONE, 0xFFu, 0.1, 100.0, origin, dir));

    output[0] = rayQueryGetCurrentTMax(&rq);

    while rayQueryProceed(&rq) {
        if hit_t <= rayQueryGetCurrentTMax(&rq) {
            rayQueryGenerateIntersection(&rq, hit_t);
        }
    }

    return rayQueryGetCurrentTMax(&rq);
}

/// Traces one ray and stores the result.
@compute @workgroup_size(1)
fn main() {
    output[1] = closest_hit_t(vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 2.0);
}
