struct RWStructuredBuffer {
    member: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> data: RWStructuredBuffer;

fn main_1() {
    var phi_19_: u32;

    phi_19_ = 0u;
    loop {
        let _e7 = phi_19_;
        switch 0i {
            default: {
                if (_e7 < 4u) {
                } else {
                    break;
                }
                let _e11 = data.member[_e7];
                if (_e11 == 1u) {
                    break;
                }
                let _e13 = data.member[_e7];
                if (_e13 == 2u) {
                    break;
                }
                data.member[_e7] = (_e7 * 2u);
                break;
            }
        }
        continue;
        continuing {
            phi_19_ = (_e7 + 1u);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
}
