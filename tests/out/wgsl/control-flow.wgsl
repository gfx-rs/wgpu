fn switch_default_break(i: i32) {
    switch i {
        default: {
            break;
        }
    }
}

fn switch_case_break() {
    switch 0 {
        case 0: {
            break;
        }
        default: {
        }
    }
    return;
}

fn loop_switch_continue(x: i32) {
    loop {
        switch x {
            case 1: {
                continue;
            }
            default: {
            }
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pos: i32;

    storageBarrier();
    workgroupBarrier();
    switch 1 {
        default: {
            pos = 1;
        }
    }
    let _e4 = pos;
    switch _e4 {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
        }
        case 3: {
            pos = 2;
            fallthrough;
        }
        case 4: {
            pos = 3;
            fallthrough;
        }
        default: {
            pos = 4;
        }
    }
    switch 0u {
        case 0u: {
        }
        default: {
        }
    }
    let _e11 = pos;
    switch _e11 {
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
