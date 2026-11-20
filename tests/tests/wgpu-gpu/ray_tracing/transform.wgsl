enable wgpu_ray_query;

struct Probe {
    origin: vec3f,
    flags: u32,
    direction: vec3f,
    padding: u32,
}

struct Hit {
    kind: u32,
    t: f32,
    front_face: u32,
}

@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage> probes: array<Probe>;
@group(0) @binding(2) var<storage, read_write> hits: array<Hit>;

@compute @workgroup_size(1)
fn trace(@builtin(global_invocation_id) id: vec3u) {
    let probe = probes[id.x];
    var query: ray_query;
    rayQueryInitialize(&query, scene, RayDesc(probe.flags, 0xFFu, 0.0, 100.0, probe.origin, probe.direction));
    // The BLAS geometry is opaque, so every triangle hit is committed by proceeding.
    while rayQueryProceed(&query) {}
    let committed = rayQueryGetCommittedIntersection(&query);
    hits[id.x] = Hit(committed.kind, committed.t, u32(committed.front_face));
}
