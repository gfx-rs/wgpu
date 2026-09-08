enable wgpu_ray_query;

struct Probe {
    origin: vec3f,
    flags: u32,
    direction: vec3f,
    mask: u32,
    near: f32,
    far: f32,
    accept: u32,
    padding: u32,
}
struct Hit {
    kind: u32,
    t: f32,
    custom: u32,
    instance: u32,
    geometry: u32,
    primitive: u32,
    barycentrics: vec2f,
}
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage> probes: array<Probe>;
@group(0) @binding(2) var<storage, read_write> hits: array<Hit>;

fn trace(index: u32) -> Hit {
    let p = probes[index];
    var query: ray_query;
    rayQueryInitialize(&query, scene, RayDesc(p.flags, p.mask, p.near, p.far, p.origin, p.direction));
    while rayQueryProceed(&query) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        if p.accept != 0u {
            if candidate.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
                rayQueryConfirmIntersection(&query);
            } else if candidate.kind == RAY_QUERY_INTERSECTION_AABB {
                // This probe passes through the sphere centre along its z axis.
                let t = (2.0 - p.origin.z) / p.direction.z;
                if t >= p.near && t <= p.far {
                    rayQueryGenerateIntersection(&query, t);
                }
            }
        }
    }
    let h = rayQueryGetCommittedIntersection(&query);
    var out: Hit;
    out.kind = h.kind;
    if h.kind != 0u {
        out = Hit(h.kind, h.t, h.instance_custom_data, h.instance_index,
            h.geometry_index, h.primitive_index, h.barycentrics);
    }
    return out;
}

@compute @workgroup_size(1)
fn compute(@builtin(global_invocation_id) id: vec3u) {
    hits[id.x] = trace(id.x);
}
struct VertexHit {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) hit: vec4u,
}
@vertex
fn vertex_query(@builtin(vertex_index) index: u32) -> VertexHit {
    let h = trace(0u);
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return VertexHit(vec4f(p[index], 0.0, 1.0), vec4u(h.kind, bitcast<u32>(h.t), h.custom, h.instance));
}
@fragment
fn fragment_vertex(input: VertexHit) -> @location(0) vec4u {
    return input.hit;
}


@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(p[index], 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4u {
    let h = trace(u32(p.x));
    return vec4u(h.kind, bitcast<u32>(h.t), h.custom, h.instance);
}
