// Subgroup operations inside a `switch`.
//
// The MSL backend lowers these to nested `if`/`else` statements, because
// Apple's Metal compiler miscompiles subgroup operations performed in a
// `switch` case. The remaining functions cover the cases where that lowering
// must not be applied.

// Lowered: no fall-through, no `break` bound to the switch.
fn lowered(sid: u32) -> u32 {
    var v = 0u;
    switch sid % 3u {
        case 0u: { v = subgroupBroadcastFirst(sid); }
        case 1u: { v = subgroupAdd(sid); }
        default: { v = subgroupBroadcastFirst(sid); }
    }
    return v;
}

// Not lowered: a case falls through, which an if/else chain cannot express.
fn kept_fallthrough(sid: u32) -> u32 {
    var v = 0u;
    switch sid % 2u {
        case 0u, default: { v = subgroupBroadcastFirst(sid); }
    }
    return v;
}

// Not lowered: the `break` binds to the switch, and would bind to the
// enclosing loop if this became an if/else chain.
fn kept_break(sid: u32) -> u32 {
    var v = 0u;
    loop {
        switch sid % 2u {
            case 0u: {
                v = subgroupBroadcastFirst(sid);
                break;
            }
            default: { v = 1u; }
        }
        break;
    }
    return v;
}

// Not lowered: no subgroup operations in any case body.
fn kept_no_subgroup_ops(sid: u32) -> u32 {
    var v = 0u;
    switch sid % 2u {
        case 0u: { v = 10u; }
        default: { v = 20u; }
    }
    return v;
}

@compute @workgroup_size(32)
fn main(@builtin(subgroup_invocation_id) sid: u32) {
    _ = lowered(sid);
    _ = kept_fallthrough(sid);
    _ = kept_break(sid);
    _ = kept_no_subgroup_ops(sid);
}
