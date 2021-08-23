[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    var pos: i32;

    storageBarrier();
    workgroupBarrier();
    switch(1) {
        default: {
            pos = 1;
        }
    }
    let _e4: i32 = pos;
    switch(_e4) {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            return;
        }
        case 3: {
            pos = 2;
            fallthrough;
        }
        case 4: {
            return;
        }
        default: {
            pos = 3;
            return;
        }
    }
}
