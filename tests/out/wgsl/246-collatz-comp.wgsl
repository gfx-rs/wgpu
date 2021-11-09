[[block]]
struct PrimeIndices {
    indices: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> global: PrimeIndices;
var<private> gl_GlobalInvocationID: vec3<u32>;

fn collatz_iterations(n: u32) -> u32 {
    var n_1: u32;
    var i: u32 = 0u;

    n_1 = n;
    loop {
        let e7: u32 = n_1;
        if (!((e7 != u32(1)))) {
            break;
        }
        {
            let e14: u32 = n_1;
            if (((f32(e14) % f32(2)) == f32(0))) {
                {
                    let e22: u32 = n_1;
                    n_1 = (e22 / u32(2));
                }
            } else {
                {
                    let e27: u32 = n_1;
                    n_1 = ((u32(3) * e27) + u32(1));
                }
            }
            let e33: u32 = i;
            i = (e33 + 1u);
        }
    }
    let e36: u32 = i;
    return e36;
}

fn main_1() {
    var index: u32;

    let e3: vec3<u32> = gl_GlobalInvocationID;
    index = e3.x;
    let e6: u32 = index;
    let e8: u32 = index;
    let e11: u32 = index;
    let e13: u32 = global.indices[e11];
    let e14: u32 = collatz_iterations(e13);
    global.indices[e6] = e14;
    return;
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] param: vec3<u32>) {
    gl_GlobalInvocationID = param;
    main_1();
    return;
}
