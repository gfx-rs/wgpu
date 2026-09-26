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
fn main(@builtin(local_invocation_index) local_invocation_index: u32) {
    var zero_init_index: u32;

    if (local_invocation_index < 16u) {
        small[local_invocation_index] = f32();
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e72 = zero_init_index;
        particles[_e72] = Particle();
        continuing {
            let _e78 = zero_init_index;
            let _e79 = (_e78 + 32u);
            zero_init_index = _e79;
            break if (_e79 >= 256u);
        }
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e96 = zero_init_index;
        mixed.grid[(_e96 / 18u)][(_e96 % 18u)] = vec4<f32>();
        continuing {
            let _e107 = zero_init_index;
            let _e108 = (_e107 + 32u);
            zero_init_index = _e108;
            break if (_e108 >= 324u);
        }
    }
    if (local_invocation_index < 3u) {
        atomicStore((&mixed.cells[local_invocation_index].count), u32());
    }
    zero_init_index = local_invocation_index;
    loop {
        let _e120 = zero_init_index;
        atomicStore((&mixed.cells[(_e120 / 100u)].bins[(_e120 % 100u)]), u32());
        continuing {
            let _e132 = zero_init_index;
            let _e133 = (_e132 + 32u);
            zero_init_index = _e133;
            break if (_e133 >= 300u);
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
    let _e3 = scalar;
    let _e7 = small[3];
    let _e13 = particles[7].vel.y;
    let _e18 = single[0].x;
    output[0] = (((f32(_e3) + _e7) + _e13) + f32(_e18));
    let _e27 = mixed.transform[1][0];
    let _e30 = mixed.flag;
    let _e38 = mixed.grid[2][5].z;
    output[1] = ((_e27 + f32(_e30)) + _e38);
    let _e47 = atomicLoad((&mixed.cells[1].bins[42]));
    let _e52 = atomicLoad((&mixed.cells[2].count));
    output[2] = f32((_e47 + _e52));
    let _e58 = atomicLoad((&counter));
    output[3] = f32(_e58);
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
