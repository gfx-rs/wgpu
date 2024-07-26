const SIZE: u32 = 128u;

@group(0) @binding(0) 
var<storage, read_write> arr_i64_: array<atomic<i64>, 128>;
@group(0) @binding(1) 
var<storage, read_write> arr_u64_: array<atomic<u64>, 128>;

@compute @workgroup_size(1, 1, 1) 
fn test_atomic_compare_exchange_i64_() {
    var i: u32 = 0u;
    var old: i64;
    var exchanged: bool;

    loop {
        let _e2 = i;
        if (_e2 < SIZE) {
        } else {
            break;
        }
        {
            let _e6 = i;
            let _e8 = atomicLoad((&arr_i64_[_e6]));
            old = _e8;
            exchanged = false;
            loop {
                let _e12 = exchanged;
                if !(_e12) {
                } else {
                    break;
                }
                {
                    let _e14 = old;
                    let new_ = bitcast<i64>((_e14 + 10li));
                    let _e19 = i;
                    let _e21 = old;
                    let _e22 = atomicCompareExchangeWeak((&arr_i64_[_e19]), _e21, new_);
                    old = _e22.old_value;
                    exchanged = _e22.exchanged;
                }
            }
        }
        continuing {
            let _e26 = i;
            i = (_e26 + 1u);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn test_atomic_compare_exchange_u64_() {
    var i_1: u32 = 0u;
    var old_1: u64;
    var exchanged_1: bool;

    loop {
        let _e2 = i_1;
        if (_e2 < SIZE) {
        } else {
            break;
        }
        {
            let _e6 = i_1;
            let _e8 = atomicLoad((&arr_u64_[_e6]));
            old_1 = _e8;
            exchanged_1 = false;
            loop {
                let _e12 = exchanged_1;
                if !(_e12) {
                } else {
                    break;
                }
                {
                    let _e14 = old_1;
                    let new_1 = bitcast<u64>((_e14 + 10lu));
                    let _e19 = i_1;
                    let _e21 = old_1;
                    let _e22 = atomicCompareExchangeWeak((&arr_u64_[_e19]), _e21, new_1);
                    old_1 = _e22.old_value;
                    exchanged_1 = _e22.exchanged;
                }
            }
        }
        continuing {
            let _e26 = i_1;
            i_1 = (_e26 + 1u);
        }
    }
    return;
}
