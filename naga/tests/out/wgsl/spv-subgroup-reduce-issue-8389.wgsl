var<private> global: u32;

fn function_() {
    var phi_16_: u32;
    var local: u32;

    let _e4 = global;
    phi_16_ = 0u;
    loop {
        let _e6 = phi_16_;
        local = (_e4 + _e6);
        continue;
        continuing {
            let _e8 = (_e6 + 1u);
            phi_16_ = _e8;
            break if !((_e8 < 4u));
        }
    }
    let _e12 = local;
    let _e13 = subgroupMin(_e12);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_index) param: u32) {
    global = param;
    function_();
}
