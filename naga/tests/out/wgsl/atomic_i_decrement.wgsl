struct type_3 {
    member: array<u32>,
}

struct type_5 {
    member: atomic<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_5;
@group(0) @binding(1) 
var<storage, read_write> global_1: type_3;

fn function() {
    var phi_40_: bool;

    loop {
        let _e8 = atomicSub((&global.member), 1u);
        if (_e8 < arrayLength((&global_1.member))) {
            global_1.member[_e8] = _e8;
            phi_40_ = select(true, false, (_e8 == 0u));
        } else {
            phi_40_ = false;
        }
        let _e16 = phi_40_;
        continue;
        continuing {
            break if !(_e16);
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_i_decrement() {
    function();
}
