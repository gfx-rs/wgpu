[[block]]
struct PrimeIndices {
    data: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]]
var<storage> v_indices: [[access(read_write)]] PrimeIndices;

fn collatz_iterations(n_base: u32) -> u32 {
    var n: u32;
    var i: u32 = 0u;

    n = n_base;
    loop {
        let _e5: u32 = n;
        if ((_e5 <= 1u)) {
            break;
        }
        let _e8: u32 = n;
        if (((_e8 % 2u) == 0u)) {
            let _e13: u32 = n;
            n = (_e13 / 2u);
        } else {
            let _e17: u32 = n;
            n = ((3u * _e17) + 1u);
        }
        let _e21: u32 = i;
        i = (_e21 + 1u);
    }
    let _e24: u32 = i;
    return _e24;
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let _e8: u32 = v_indices.data[global_id.x];
    let _e9: u32 = collatz_iterations(_e8);
    v_indices.data[global_id.x] = _e9;
    return;
}
