struct type_2 {
    member: u32,
}

struct type_4 {
    member: atomic<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_4;
@group(0) @binding(1) 
var<storage> global_1: type_2;

fn function() {
    var phi_23_: u32;
    var phi_24_: u32;

    phi_23_ = 0u;
    loop {
        let _e10 = phi_23_;
        let _e11 = global_1.member;
        let _e12 = (_e10 >= _e11);
        if _e12 {
            phi_24_ = u32();
        } else {
            let _e13 = atomicAdd((&global.member), 1u);
            phi_24_ = (_e10 + 1u);
        }
        let _e17 = phi_24_;
        continue;
        continuing {
            phi_23_ = _e17;
            break if !(select(true, false, _e12));
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_i_increment() {
    function();
}
