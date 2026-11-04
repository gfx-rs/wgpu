enable wgpu_ray_query;

/// The scene to trace against.
@group(0) @binding(0)
var acc_struct: acceleration_structure;

/// Receives the values that the shaders read.
@group(0) @binding(1)
var<storage, read_write> output: array<f32, 5>;

/// Generates one hit, then a farther one, and records the current t max after each.
@compute @workgroup_size(1)
fn single_hit() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 100.0, vec3<f32>(0.0), vec3<f32>(0.01, 0.02, 1.0)));
    output[0] = rayQueryGetCurrentTMax(&rq);

    var iterations = 0.0;
    while rayQueryProceed(&rq) {
        iterations = iterations + 1.0;
        rayQueryGenerateIntersection(&rq, 3.0);
        output[1] = rayQueryGetCurrentTMax(&rq);
        rayQueryGenerateIntersection(&rq, 3.5);
        output[2] = rayQueryGetCurrentTMax(&rq);
    }

    output[3] = rayQueryGetCurrentTMax(&rq);
    output[4] = iterations;
}

/// Generates the near t of each AABB that the ray reaches, and records the final current t max and the AABB count.
@compute @workgroup_size(1)
fn nearest_of_two() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 100.0, vec3<f32>(0.0), vec3<f32>(0.01, 0.02, 1.0)));
    output[0] = rayQueryGetCurrentTMax(&rq);

    var iterations = 0.0;
    while rayQueryProceed(&rq) {
        iterations = iterations + 1.0;
        let candidate = rayQueryGetCandidateIntersection(&rq);
        let hit_t = select(2.0, 6.0, candidate.primitive_index == 1u);
        if hit_t <= rayQueryGetCurrentTMax(&rq) {
            rayQueryGenerateIntersection(&rq, hit_t);
        }
    }

    output[1] = rayQueryGetCurrentTMax(&rq);
    output[2] = iterations;
}
