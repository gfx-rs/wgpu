struct PrimeIndices {
    indices: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: PrimeIndices;
var<private> gl_GlobalInvocationID: vec3<u32>;

fn collatz_iterations(n: u32) -> u32 {
    var n_1: u32;
    var i: u32;

    n_1 = n;
    i = u32(0);
    loop {
        let _e7 = n_1;
        if !((_e7 != u32(1))) {
            break;
        }
        {
            let _e14 = n_1;
            let _e15 = f32(_e14);
            let _e17 = f32(2);
            if ((_e15 - (floor((_e15 / _e17)) * _e17)) == f32(0)) {
                {
                    let _e25 = n_1;
                    n_1 = (_e25 / u32(2));
                }
            } else {
                {
                    let _e30 = n_1;
                    n_1 = ((u32(3) * _e30) + u32(1));
                }
            }
            let _e36 = i;
            i = (_e36 + 1u);
        }
    }
    let _e39 = i;
    return _e39;
}

fn main_1() {
    var index: u32;

    let _e3 = gl_GlobalInvocationID;
    index = _e3.x;
    let _e6 = index;
    let _e8 = index;
    let _e11 = index;
    let _e13 = global.indices[_e11];
    let _e14 = collatz_iterations(_e13);
    global.indices[_e6] = _e14;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) param: vec3<u32>) {
    gl_GlobalInvocationID = param;
    main_1();
    return;
}
