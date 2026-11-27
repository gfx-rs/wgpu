enable wgpu_ray_query;
enable wgpu_binding_array;

struct UniformIndex {
    index: u32,
};

@group(0) @binding(0)
var tlas_array_bounded: binding_array<acceleration_structure, 4>;
@group(0) @binding(1)
var tlas_array_unbounded: binding_array<acceleration_structure>;
@group(0) @binding(2)
var<uniform> uni: UniformIndex;
@group(0) @binding(3)
var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(2)
fn main(@builtin(local_invocation_index) local_index: u32) {
    var rq: ray_query;

    // Dynamically uniform index.
    rayQueryInitialize(
        &rq,
        tlas_array_bounded[uni.index],
        RayDesc(0x04u, 0xFFu, 0.1, 100.0, vec3<f32>(0.0, 0.0, -2.0), vec3<f32>(0.0, 0.0, 1.0)),
    );
    while (rayQueryProceed(&rq)) {}
    let first = rayQueryGetCommittedIntersection(&rq);

    // Non-uniform index.
    rayQueryInitialize(
        &rq,
        tlas_array_unbounded[local_index],
        RayDesc(0x04u, 0xFFu, 0.1, 100.0, vec3<f32>(first.barycentrics, 0.0), vec3<f32>(0.0, 0.0, 1.0)),
    );
    while (rayQueryProceed(&rq)) {}
    let second = rayQueryGetCommittedIntersection(&rq);

    out[local_index] = vec2<f32>(first.t, second.t);
}
