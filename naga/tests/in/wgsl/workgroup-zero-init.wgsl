struct Particle {
    position: vec3<f32>,
    hits: atomic<u32>,
    history: array<f32, 4>,
}

// Not an array: stored by invocation 0.
var<workgroup> total: u32;
// Fewer elements than invocations: one guarded store.
var<workgroup> small: array<u32, 3>;
// One element per invocation: one unguarded store.
var<workgroup> exact: array<vec4<f32>, 8>;
// More elements than invocations: a strided loop over the flattened index.
var<workgroup> grid: array<array<f32, 4>, 5>;
// Struct members are zeroed separately, atomics with atomic stores.
var<workgroup> particles: array<Particle, 2>;

@compute @workgroup_size(4, 2)
fn main(@builtin(local_invocation_index) index: u32) {
    let sum = f32(small[index % 3u]) + exact[index].x + grid[index % 5u][index % 4u]
        + particles[index % 2u].position.x + particles[index % 2u].history[index % 4u];
    atomicAdd(&particles[0].hits, u32(sum));
    if index == 0u {
        total = atomicLoad(&particles[1].hits);
    }
}

// Without a `local_invocation_index` argument, the pass adds one.
@compute @workgroup_size(64)
fn no_index() {
    _ = total;
}
