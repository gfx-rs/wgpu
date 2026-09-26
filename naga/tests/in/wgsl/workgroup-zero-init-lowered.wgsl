struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
}

struct WithAtomics {
    count: atomic<u32>,
    bins: array<atomic<u32>, 100>,
}

struct Mixed {
    transform: mat4x4<f32>,
    flag: u32,
    grid: array<array<vec4<f32>, 18>, 18>,
    cells: array<WithAtomics, 3>,
}

var<workgroup> scalar: u32;
var<workgroup> small: array<f32, 16>;
var<workgroup> particles: array<Particle, 256>;
var<workgroup> single: array<vec2<u32>, 1>;
var<workgroup> mixed: Mixed;
var<workgroup> counter: atomic<i32>;

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 4)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
    let base = (id.x + id.y * 8u) * 4u;
    output[base] = f32(scalar) + small[3] + particles[7].vel.y + f32(single[0].x);
    output[base + 1u] = mixed.transform[1].x + f32(mixed.flag) + mixed.grid[2][5].z;
    output[base + 2u] = f32(atomicLoad(&mixed.cells[1].bins[42]) + atomicLoad(&mixed.cells[2].count));
    output[base + 3u] = f32(atomicLoad(&counter));
}

@compute @workgroup_size(64)
fn with_index(@builtin(local_invocation_index) index: u32) {
    output[index] = small[index % 16u];
}

struct Inputs {
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
}

@compute @workgroup_size(16)
fn with_struct(inputs: Inputs) {
    output[inputs.index] = particles[inputs.index].pos.x + f32(atomicLoad(&counter));
}
