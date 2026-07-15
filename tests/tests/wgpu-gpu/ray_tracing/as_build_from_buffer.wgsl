enable wgpu_ray_query;

@group(0) @binding(0) var acc_struct: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> out: vec4<u32>;
@group(0) @binding(2) var<uniform> ray_origin: vec4<f32>;

@workgroup_size(1) @compute
fn trace() {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.001, 100000.0, ray_origin.xyz, vec3f(0.0, 0.0, 1.0)));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    out = vec4<u32>(hit.kind, hit.instance_index, hit.instance_custom_data, bitcast<u32>(hit.t));
}
