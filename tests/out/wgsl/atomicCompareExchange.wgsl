const SIZE: u32 = 128u;

@group(0) @binding(0) 
var<storage, read_write> arr_i32_: array<atomic<i32>, 128>;
@group(0) @binding(1) 
var<storage, read_write> arr_u32_: array<atomic<u32>, 128>;

@compute @workgroup_size(1, 1, 1) 
fn test_atomic_compare_exchange_i32_() {
    var i: u32 = 0u;
    var old: i32;
    var exchanged: bool;

    loop {
        let _e2 = i;
        if (_e2 < SIZE) {
        } else {
            break;
        }
        {
            let _e6 = i;
            let _e8 = atomicLoad((&arr_i32_[_e6]));
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
                    let new_ = bitcast<i32>((bitcast<f32>(_e14) + 1.0));
                    let _e20 = i;
                    let _e22 = old;
                    let _e23 = atomicCompareExchangeWeak((&arr_i32_[_e20]), _e22, new_);
                    old = _e23.old_value;
                    exchanged = _e23.exchanged;
                }
            }
        }
        continuing {
            let _e27 = i;
            i = (_e27 + 1u);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn test_atomic_compare_exchange_u32_() {
    var i_1: u32 = 0u;
    var old_1: u32;
    var exchanged_1: bool;

    loop {
        let _e2 = i_1;
        if (_e2 < SIZE) {
        } else {
            break;
        }
        {
            let _e6 = i_1;
            let _e8 = atomicLoad((&arr_u32_[_e6]));
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
                    let new_1 = bitcast<u32>((bitcast<f32>(_e14) + 1.0));
                    let _e20 = i_1;
                    let _e22 = old_1;
                    let _e23 = atomicCompareExchangeWeak((&arr_u32_[_e20]), _e22, new_1);
                    old_1 = _e23.old_value;
                    exchanged_1 = _e23.exchanged;
                }
            }
        }
        continuing {
            let _e27 = i_1;
            i_1 = (_e27 + 1u);
        }
    }
    return;
}
