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
    var phi_26_: type_2;
    var phi_29_: u32;
    var phi_43_: type_2;
    var phi_44_: type_2;
    var phi_53_: bool;
    var phi_27_: type_2;
    var phi_30_: u32;

    let _e10 = global_1.member;
    phi_26_ = type_2(0u, _e10);
    phi_29_ = 0u;
    loop {
        let _e13 = phi_26_;
        let _e15 = phi_29_;
        if (_e13.member < _e13.member_1) {
            phi_43_ = type_2((_e13.member + 1u), _e13.member_1);
            phi_44_ = type_2(1u, _e13.member);
        } else {
            phi_43_ = _e13;
            phi_44_ = type_2(0u, type_2().member_1);
        }
        let _e26 = phi_43_;
        let _e28 = phi_44_;
        switch bitcast<i32>(_e28.member) {
            case 0: {
                phi_53_ = false;
                phi_27_ = type_2();
                phi_30_ = u32();
                break;
            }
            case 1: {
                let _e31 = atomicExchange((&global.member), _e15);
                phi_53_ = true;
                phi_27_ = _e26;
                phi_30_ = (_e15 + _e31);
                break;
            }
            default: {
                phi_53_ = false;
                phi_27_ = type_2();
                phi_30_ = u32();
                break;
            }
        }
        let _e34 = phi_53_;
        let _e36 = phi_27_;
        let _e38 = phi_30_;
        continue;
        continuing {
            phi_26_ = _e36;
            phi_29_ = _e38;
            break if !(_e34);
        }
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_exchange() {
    function();
}
