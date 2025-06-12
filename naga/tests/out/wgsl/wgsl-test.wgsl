@group(0) @binding(0) 
var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var i: u32 = 0u;

    loop {
        let _e2 = i;
        if (_e2 < 4u) {
        } else {
            break;
        }
        {
            let _e6 = i;
            let _e8 = i;
            data[_e6] = (_e8 * 2u);
        }
        continuing {
            let _e12 = i;
            i = (_e12 + 1u);
        }
    }
    return;
}
