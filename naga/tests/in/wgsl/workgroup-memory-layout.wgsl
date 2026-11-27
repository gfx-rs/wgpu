override N: u32 = 513;
const W: u32 = 64u;

struct Padded {
    @align(16) xyz: vec3u,
    tail: u32,
}

var<workgroup> flag: u32;
var<workgroup> pair: vec2u;
var<workgroup> scratch: array<u32, N>;
var<workgroup> counter: atomic<u32>;
var<workgroup> vectors: array<vec3u, 3>;
var<workgroup> padded: Padded;

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

fn sum_small() -> u32 {
    var sum = flag + pair.x + pair.y + padded.xyz.x + padded.xyz.y + padded.xyz.z + padded.tail;
    for (var j = 0u; j < 3u; j++) {
        sum += vectors[j].x + vectors[j].y + vectors[j].z;
    }
    return sum;
}

@compute @workgroup_size(W)
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) group: vec3u) {
    let base = group.x * 4u;
    if i == 0u {
        var initial = sum_small() + atomicLoad(&counter);
        for (var j = 0u; j < N; j++) {
            initial += scratch[j];
        }
        output[base] = initial;
    }
    workgroupBarrier();

    for (var j = i; j < N; j += W) {
        scratch[j] = j + group.x;
    }
    atomicAdd(&counter, 1u);
    if i == 0u {
        flag = 1u;
        pair = vec2u(2u, 3u);
        vectors = array(vec3u(4u, 5u, 6u), vec3u(7u, 8u, 9u), vec3u(10u, 11u, 12u));
        padded = Padded(vec3u(13u, 14u, 15u), 16u);
    }
    workgroupBarrier();

    if i == 0u {
        var sum = 0u;
        for (var j = 0u; j < N; j++) {
            sum += scratch[j];
        }
        output[base + 1u] = sum;
        output[base + 2u] = atomicLoad(&counter);
        output[base + 3u] = sum_small();
    }
}

@compute @workgroup_size(1)
fn padded_only(@builtin(workgroup_id) group: vec3u) {
    let base = group.x * 4u;
    output[base] = padded.xyz.x + padded.xyz.y + padded.xyz.z + padded.tail;
    padded = Padded(vec3u(13u, 14u, 15u), 16u);
    output[base + 1u] = padded.xyz.x + padded.xyz.y + padded.xyz.z + padded.tail;
    output[base + 2u] = 0u;
    output[base + 3u] = 0u;
}

@compute @workgroup_size(1)
fn no_workgroup(@builtin(workgroup_id) group: vec3u) {
    let base = group.x * 4u;
    output[base] = group.x;
    output[base + 1u] = 0u;
    output[base + 2u] = 0u;
    output[base + 3u] = 0u;
}
