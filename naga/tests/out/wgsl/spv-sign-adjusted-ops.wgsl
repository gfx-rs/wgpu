struct type_2 {
    member: u32,
    member_1: i32,
    member_2: u32,
    member_3: i32,
}

@group(0) @binding(0)
var<storage, read_write> global: type_2;

fn function_() {
    let _e5 = global.member;
    let _e6 = global.member_1;
    let _e7 = global.member_2;
    let _e8 = global.member_3;
    global.member_1 = -(i32(_e5));
    global.member_2 = (_e5 + u32(_e6));
    global.member = (_e7 << u32(_e8));
    global.member = extractBits(_e7, u32(_e6), u32(_e8));
    global.member_2 = insertBits(_e7, _e5, u32(_e6), u32(_e8));
    switch i32(_e5) {
        case 1: {
            break;
        }
        default: {
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    function_();
}
