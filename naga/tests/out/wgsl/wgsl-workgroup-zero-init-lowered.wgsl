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

struct Inputs {
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
}

var<workgroup> scalar: u32;
var<workgroup> small: array<f32, 16>;
var<workgroup> particles: array<Particle, 256>;
var<workgroup> single: array<vec2<u32>, 1>;
var<workgroup> mixed: Mixed;
var<workgroup> counter: atomic<i32>;
@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 4, 1)
fn main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    var zero_init_index: u32;

    if (local_invocation_index < 16u) {
        small[local_invocation_index] = f32();
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e86 = zero_init_index;
        particles[_e86] = Particle();
        continuing {
            let _e92 = zero_init_index;
            let _e93 = (_e92 + 32u);
            zero_init_index = _e93;
            break if (_e93 >= 256u);
        }
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e110 = zero_init_index;
        mixed.grid[(_e110 / 18u)][(_e110 % 18u)] = vec4<f32>();
        continuing {
            let _e121 = zero_init_index;
            let _e122 = (_e121 + 32u);
            zero_init_index = _e122;
            break if (_e122 >= 324u);
        }
    }
    if (local_invocation_index < 3u) {
        atomicStore((&mixed.cells[local_invocation_index].count), u32());
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e134 = zero_init_index;
        atomicStore((&mixed.cells[(_e134 / 100u)].bins[(_e134 % 100u)]), u32());
        continuing {
            let _e146 = zero_init_index;
            let _e147 = (_e146 + 32u);
            zero_init_index = _e147;
            break if (_e147 >= 300u);
        }
    }
    if (local_invocation_index == 0u) {
        scalar = u32();
        single[0u] = vec2<u32>();
        mixed.transform = mat4x4<f32>();
        mixed.flag = u32();
        atomicStore((&counter), i32());
    }
    workgroupBarrier();
    let base = ((id.x + (id.y * 8u)) * 4u);
    let _e11 = scalar;
    let _e15 = small[3];
    let _e21 = particles[7].vel.y;
    let _e26 = single[0].x;
    output[base] = (((f32(_e11) + _e15) + _e21) + f32(_e26));
    let _e37 = mixed.transform[1][0];
    let _e40 = mixed.flag;
    let _e48 = mixed.grid[2][5].z;
    output[(base + 1u)] = ((_e37 + f32(_e40)) + _e48);
    let _e59 = atomicLoad((&mixed.cells[1].bins[42]));
    let _e64 = atomicLoad((&mixed.cells[2].count));
    output[(base + 2u)] = f32((_e59 + _e64));
    let _e72 = atomicLoad((&counter));
    output[(base + 3u)] = f32(_e72);
    return;
}

@compute @workgroup_size(64, 1, 1)
fn with_index(@builtin(local_invocation_index) index: u32) {
    if (index < 16u) {
        small[index] = f32();
    }
    workgroupBarrier();
    let _e7 = small[(index % 16u)];
    output[index] = _e7;
    return;
}

@compute @workgroup_size(16, 1, 1)
fn with_struct(inputs: Inputs) {
    var zero_init_index_1: u32;

    zero_init_index_1 = inputs.index;
    loop {
        let _e18 = zero_init_index_1;
        particles[_e18] = Particle();
        continuing {
            let _e24 = zero_init_index_1;
            let _e25 = (_e24 + 16u);
            zero_init_index_1 = _e25;
            break if (_e25 >= 256u);
        }
    }
    if (inputs.index == 0u) {
        atomicStore((&counter), i32());
    }
    workgroupBarrier();
    let _e9 = particles[inputs.index].pos.x;
    let _e11 = atomicLoad((&counter));
    output[inputs.index] = (_e9 + f32(_e11));
    return;
}
