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
        let _e4 = n_1;
        if !((_e4 != 1u)) {
            break;
        }
        {
            let _e8 = n_1;
            let _e9 = f32(_e8);
            if ((_e9 - (floor((_e9 / 2f)) * 2f)) == 0f) {
                {
                    let _e17 = n_1;
                    n_1 = (_e17 / 2u);
                }
            } else {
                {
                    let _e20 = n_1;
                    n_1 = ((3u * _e20) + 1u);
                }
            }
            let _e25 = i;
            i = (_e25 + 1u);
        }
    }
    let _e28 = i;
    return _e28;
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
