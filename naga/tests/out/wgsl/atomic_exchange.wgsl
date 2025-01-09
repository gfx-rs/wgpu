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
    var phi_36_: u32;
    var phi_52_: type_2;
    var phi_53_: type_2;
    var phi_62_: bool;
    var phi_34_: type_2;
    var phi_37_: u32;

    let _e10 = global_1.member;
    phi_33_ = type_2(0u, _e10);
    phi_36_ = 0u;
    loop {
        let _e13 = phi_33_;
        let _e15 = phi_36_;
        if (_e13.member < _e13.member_1) {
            phi_52_ = type_2((_e13.member + 1u), _e13.member_1);
            phi_53_ = type_2(1u, _e13.member);
        } else {
            phi_52_ = _e13;
            phi_53_ = type_2(0u, type_2().member_1);
        }
        let _e26 = phi_52_;
        let _e28 = phi_53_;
        switch bitcast<i32>(_e28.member) {
            case 0: {
                phi_62_ = false;
                phi_34_ = type_2();
                phi_37_ = u32();
                break;
            }
            case 1: {
                let _e31 = atomicExchange((&global.member), _e15);
                phi_62_ = true;
                phi_34_ = _e26;
                phi_37_ = (_e15 + _e31);
                break;
            }
            default: {
                phi_62_ = false;
                phi_34_ = type_2();
                phi_37_ = u32();
                break;
            }
        }
        let _e34 = phi_62_;
        let _e36 = phi_34_;
        let _e38 = phi_37_;
        continue;
        continuing {
            phi_33_ = _e36;
            phi_36_ = _e38;
            break if !(_e34);
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_exchange() {
    function();
}
