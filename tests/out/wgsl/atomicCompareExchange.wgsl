struct gen___atomic_compare_exchange_result {
    old_value: i32,
    exchanged: bool,
}

struct gen___atomic_compare_exchange_result_1 {
    old_value: u32,
    exchanged: bool,
}

let SIZE: u32 = 128u;

@group(0) @binding(0) 
var<storage, read_write> arr_i32_: array<atomic<i32>,SIZE>;
@group(0) @binding(1) 
var<storage, read_write> arr_u32_: array<atomic<u32>,SIZE>;

@compute @workgroup_size(1, 1, 1) 
fn test_atomic_compare_exchange_i32_() {
    var i: u32 = 0u;
    var old: i32;
    var exchanged: bool;

    loop {
        let _e5 = i;
        if (_e5 < SIZE) {
        } else {
            break;
        }
        let _e10 = i;
        let _e12 = atomicLoad((&arr_i32_[_e10]));
        old = _e12;
        exchanged = false;
        loop {
            let _e16 = exchanged;
            if !(_e16) {
            } else {
                break;
            }
            let _e18 = old;
            let new_ = bitcast<i32>((bitcast<f32>(_e18) + 1.0));
            let _e23 = i;
            let _e25 = old;
            let _e26 = atomicCompareExchangeWeak((&arr_i32_[_e23]), _e25, new_);
            old = _e26.old_value;
            exchanged = _e26.exchanged;
        }
        continuing {
            let _e7 = i;
            i = (_e7 + 1u);
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
        let _e5 = i_1;
        if (_e5 < SIZE) {
        } else {
            break;
        }
        let _e10 = i_1;
        let _e12 = atomicLoad((&arr_u32_[_e10]));
        old_1 = _e12;
        exchanged_1 = false;
        loop {
            let _e16 = exchanged_1;
            if !(_e16) {
            } else {
                break;
            }
            let _e18 = old_1;
            let new_1 = bitcast<u32>((bitcast<f32>(_e18) + 1.0));
            let _e23 = i_1;
            let _e25 = old_1;
            let _e26 = atomicCompareExchangeWeak((&arr_u32_[_e23]), _e25, new_1);
            old_1 = _e26.old_value;
            exchanged_1 = _e26.exchanged;
        }
        continuing {
            let _e7 = i_1;
            i_1 = (_e7 + 1u);
        }
    }
    return;
}
