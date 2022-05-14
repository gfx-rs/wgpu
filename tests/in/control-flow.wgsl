@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //TODO: execution-only barrier?
    storageBarrier();
    workgroupBarrier();

    var pos: i32;
    // switch without cases
    switch 1 {
        default: {
            pos = 1;
        }
    }

    // non-empty switch *not* in last-statement-in-function position
    // (return statements might be inserted into the switch cases otherwise)
    switch pos {
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

	// switch with unsigned integer selectors
	switch(0u) {
		case 0u: {
		}
        default: {
        }
	}

    // non-empty switch in last-statement-in-function position
    switch pos {
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
        case 4: {}
        default: {
            pos = 3;
        }
    }
}

fn switch_default_break(i: i32) {
    switch i {
        default: {
            break;
        }
    }
}

fn switch_case_break() {
    switch(0) {
        case 0: {
            break;
        }
        default: {}
    }
    return;
}

fn loop_switch_continue(x: i32) {
    loop {
        switch x {
            case 1: {
                continue;
            }
            default: {}
        }
    }
}
