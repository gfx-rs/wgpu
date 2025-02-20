struct PrimeIndices {
    data: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> v_indices: PrimeIndices;

fn collatz_iterations(n_base: u32) -> u32 {
    var n: u32;
    var i: u32 = 0u;

    n = n_base;
    loop {
        let _e4 = n;
        if (_e4 > 1u) {
        } else {
            break;
        }
        {
            let _e7 = n;
            if ((_e7 % 2u) == 0u) {
                let _e12 = n;
                n = (_e12 / 2u);
            } else {
                let _e16 = n;
                n = ((3u * _e16) + 1u);
            }
            let _e20 = i;
            i = (_e20 + 1u);
        }
    }
    let _e23 = i;
    return _e23;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let _e9 = v_indices.data[global_id.x];
    let _e10 = collatz_iterations(_e9);
    v_indices.data[global_id.x] = _e10;
    return;
}
