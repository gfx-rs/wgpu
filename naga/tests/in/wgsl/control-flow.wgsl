fn control_flow() {
    //TODO: execution-only barrier?
    storageBarrier();
    workgroupBarrier();
    textureBarrier();

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
        case 3, 4: {
            pos = 2;
        }
        case 5: {
            pos = 3;
        }
        case default, 6: {
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

fn switch_selector_type_conversion() {
    switch (0u) {
        case 0: {
        }
        default: {
        }
    }

    switch (0) {
        case 0u: {
        }
        default: {
        }
    }
}

const ONE = 1;
fn switch_const_expr_case_selectors() {
    const TWO = 2;
    switch (0) {
        case i32(): {
        }
        case ONE: {
        }
        case TWO: {
        }
        case 1 + 2: {
        }
        case vec4(4).x: {
        }
        default: {
        }
    }
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

fn loop_switch_continue_nesting(x: i32, y: i32, z: i32) {
    loop {
        switch x {
            case 1: {
                continue;
            }
            case 2: {
                switch y {
                    case 1: {
                        continue;
                    }
                    default: {
                        loop {
                            switch z {
                                case 1: {
                                    continue;
                                }
                                default: {}
                            }
                        }
                    }
                }
            }
            default: {}
        }


        // Degenerate switch with continue
        switch y {
            default: {
                continue;
            }
        }
    }

    // In separate loop to avoid spv validation error:
    // See https://github.com/gfx-rs/wgpu/issues/5658
    loop {
        // Nested degenerate switch with continue
        switch y {
            case 1, default: {
                switch z {
                    default: {
                        continue;
                    }
                }
            }
        }
    }
}

// Cases with some of the loop nested switches not containing continues.
// See `continue_forward` module in `naga`.
fn loop_switch_omit_continue_variable_checks(x: i32, y: i32, z: i32, w: i32) {
    // switch in loop with no continues, we expect checks after the switch
    // statement to not be generated
    var pos: i32 = 0;
    loop {
        switch x {
            case 1: {
                pos = 1;
            }
            default: {}
        }
        // check here can be omitted
    }

    loop {
        switch x {
            case 1: {}
            case 2: {
                switch y {
                    case 1: {
                        continue;
                    }
                    default: {
                        switch z {
                            case 1: {
                                pos = 2;
                            }
                            default: {}
                        }
                        // check here can be omitted
                    }
                }
                // check needs to be generated here
            }
            default: {}
        }
        // check needs to be generated here
    }
}

@compute @workgroup_size(1)
fn main() {
    control_flow();
    switch_default_break(1);
    switch_case_break();
    switch_selector_type_conversion();
    switch_const_expr_case_selectors();
    loop_switch_continue(1);
    loop_switch_continue_nesting(1, 2, 3);
    loop_switch_omit_continue_variable_checks(1, 2, 3, 4);
}
