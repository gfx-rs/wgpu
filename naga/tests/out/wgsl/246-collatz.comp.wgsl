struct PrimeIndices {
    indices: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: PrimeIndices;
var<private> gl_GlobalInvocationID_1: vec3<u32>;

fn collatz_iterations(n: u32) -> u32 {
    var n_1: u32;
    var i: u32 = 0u;

    n_1 = n;
    loop {
        let _e7 = n_1;
        if !((_e7 != 1u)) {
            break;
        }
        {
            let _e12 = n_1;
            let _e14 = f32(_e12);
            if ((_e14 - (floor((_e14 / 2f)) * 2f)) == 0f) {
                {
                    let _e23 = n_1;
                    n_1 = (_e23 / 2u);
                }
            } else {
                {
                    let _e28 = n_1;
                    n_1 = ((3u * _e28) + 1u);
                }
            }
            let _e34 = i;
            i = (_e34 + 1u);
        }
    }
    let _e37 = i;
    return _e37;
}

fn main_1() {
    var index: u32;

    let _e3 = gl_GlobalInvocationID_1;
    index = _e3.x;
    let _e6 = index;
    let _e8 = index;
    let _e10 = global.indices[_e8];
    let _e11 = collatz_iterations(_e10);
    global.indices[_e6] = _e11;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>) {
    gl_GlobalInvocationID_1 = gl_GlobalInvocationID;
    main_1();
    return;
}
