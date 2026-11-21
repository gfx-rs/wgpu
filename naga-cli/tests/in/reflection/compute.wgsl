// Storage buffers with runtime-sized arrays, workgroup memory, and overrides.

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    age: f32,
}

struct Particles {
    count: atomic<u32>,
    items: array<Particle>,
}

struct Params {
    dt: f32,
    gravity: vec3<f32>,
}

@id(0) override block_size: u32 = 64;
override damping: f32;
override use_gravity: bool = true;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: Particles;
@group(0) @binding(2) var<storage, read> seeds: array<vec4<u32>>;

var<workgroup> shared_positions: array<vec3<f32>, 64>;
var<workgroup> shared_count: atomic<u32>;

@compute @workgroup_size(block_size, 1, 1)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let p = particles.items[id.x];
    shared_positions[local] = p.position;
    atomicAdd(&shared_count, 1u);
    workgroupBarrier();
    var v = p.velocity * damping;
    if use_gravity {
        v += params.gravity * params.dt;
    }
    particles.items[id.x].velocity = v + vec3<f32>(seeds[id.x % 4].xyz);
}

@compute @workgroup_size(8, 8)
fn reset() {
    atomicStore(&particles.count, 0u);
}
