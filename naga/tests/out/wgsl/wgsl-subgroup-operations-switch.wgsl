fn lowered(sid_1: u32) -> u32 {
    var v: u32 = 0u;

    switch (sid_1 % 3u) {
        case 0u: {
            let _e5 = subgroupBroadcastFirst(sid_1);
            v = _e5;
        }
        case 1u: {
            let _e6 = subgroupAdd(sid_1);
            v = _e6;
        }
        default: {
            let _e7 = subgroupBroadcastFirst(sid_1);
            v = _e7;
        }
    }
    let _e8 = v;
    return _e8;
}

fn kept_fallthrough(sid_2: u32) -> u32 {
    var v_1: u32 = 0u;

    switch (sid_2 % 2u) {
        case 0u, default: {
            let _e5 = subgroupBroadcastFirst(sid_2);
            v_1 = _e5;
        }
    }
    let _e6 = v_1;
    return _e6;
}

fn kept_break(sid_3: u32) -> u32 {
    var v_2: u32 = 0u;

    loop {
        switch (sid_3 % 2u) {
            case 0u: {
                let _e5 = subgroupBroadcastFirst(sid_3);
                v_2 = _e5;
                break;
            }
            default: {
                v_2 = 1u;
            }
        }
        break;
    }
    let _e7 = v_2;
    return _e7;
}

fn kept_no_subgroup_ops(sid_4: u32) -> u32 {
    var v_3: u32 = 0u;

    switch (sid_4 % 2u) {
        case 0u: {
            v_3 = 10u;
        }
        default: {
            v_3 = 20u;
        }
    }
    let _e7 = v_3;
    return _e7;
}

@compute @workgroup_size(32, 1, 1) 
fn main(@builtin(subgroup_invocation_id) sid: u32) {
    let _e1 = lowered(sid);
    let _e2 = kept_fallthrough(sid);
    let _e3 = kept_break(sid);
    let _e4 = kept_no_subgroup_ops(sid);
    return;
}
