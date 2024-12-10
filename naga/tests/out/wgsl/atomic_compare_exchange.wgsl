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
    var phi_33_: type_2;
    var phi_34_: type_2;
    var phi_49_: type_2;
    var phi_63_: bool;

    let _e11 = global_1.member;
    phi_33_ = type_2(0u, _e11);
    loop {
        let _e14 = phi_33_;
        if (_e14.member < _e14.member_1) {
            phi_34_ = type_2((_e14.member + 1u), _e14.member_1);
            phi_49_ = type_2(1u, _e14.member);
        } else {
            phi_34_ = _e14;
            phi_49_ = type_2(0u, type_2().member_1);
        }
        let _e25 = phi_34_;
        let _e27 = phi_49_;
        switch bitcast<i32>(_e27.member) {
            case 0: {
                phi_63_ = false;
                break;
            }
            case 1: {
                let _e31 = atomicCompareExchangeWeak((&global.member), 3u, _e27.member_1);
                phi_63_ = select(true, false, (_e31.old_value == 3u));
                break;
            }
            default: {
                phi_63_ = bool();
                break;
            }
        }
        let _e36 = phi_63_;
        continue;
        continuing {
            phi_33_ = _e25;
            break if !(_e36);
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_compare_exchange() {
    function();
}
