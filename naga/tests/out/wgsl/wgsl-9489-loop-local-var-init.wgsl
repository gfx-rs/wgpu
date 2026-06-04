@group(0) @binding(0) 
var<storage> input: array<f32, 64>;
@group(0) @binding(1) 
var<storage, read_write> output: array<f32, 8>;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var t: u32 = 0u;
    var acc_noinit: vec4<f32>;
    var acc_init: vec4<f32>;
    var d: u32;

    loop {
        let _e2 = t;
        if (_e2 < 4u) {
        } else {
            break;
        }
        {
            acc_noinit = vec4<f32>();
            acc_init = vec4<f32>();
            d = 0u;
            loop {
                let _e11 = d;
                if (_e11 < 16u) {
                } else {
                    break;
                }
                {
                    let _e15 = t;
                    let _e18 = d;
                    let _e21 = input[((_e15 * 16u) + _e18)];
                    let v = vec4(_e21);
                    let _e23 = acc_noinit;
                    acc_noinit = (_e23 + v);
                    let _e25 = acc_init;
                    acc_init = (_e25 + v);
                }
                continuing {
                    let _e28 = d;
                    d = (_e28 + 1u);
                }
            }
            let _e31 = t;
            let _e36 = acc_noinit.x;
            output[(_e31 * 2u)] = _e36;
            let _e38 = t;
            let _e45 = acc_init.x;
            output[((_e38 * 2u) + 1u)] = _e45;
        }
        continuing {
            let _e47 = t;
            t = (_e47 + 1u);
        }
    }
    return;
}
