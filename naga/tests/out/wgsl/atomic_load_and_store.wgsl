struct type_2 {
    member: u32,
    member_1: u32,
}

struct type_3 {
    member: u32,
}

struct type_5 {
    member: atomic<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_5;
@group(0) @binding(1) 
var<storage> global_1: type_3;

fn function() {
    var phi_25_: type_2;
    var phi_40_: type_2;
    var phi_41_: type_2;
    var phi_50_: bool;
    var phi_26_: type_2;

    let _e10 = global_1.member;
    phi_25_ = type_2(0u, _e10);
    loop {
        let _e13 = phi_25_;
        if (_e13.member < _e13.member_1) {
            phi_40_ = type_2((_e13.member + 1u), _e13.member_1);
            phi_41_ = type_2(1u, _e13.member);
        } else {
            phi_40_ = _e13;
            phi_41_ = type_2(0u, type_2().member_1);
        }
        let _e24 = phi_40_;
        let _e26 = phi_41_;
        switch bitcast<i32>(_e26.member) {
            case 0: {
                phi_50_ = false;
                phi_26_ = type_2();
                break;
            }
            case 1: {
                let _e29 = atomicLoad((&global.member));
                atomicStore((&global.member), (_e29 + 2u));
                phi_50_ = true;
                phi_26_ = _e24;
                break;
            }
            default: {
                phi_50_ = false;
                phi_26_ = type_2();
                break;
            }
        }
        let _e32 = phi_50_;
        let _e34 = phi_26_;
        continue;
        continuing {
            phi_25_ = _e34;
            break if !(_e32);
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_load_and_store() {
    function();
}
