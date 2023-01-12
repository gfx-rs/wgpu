const SIZE: u32 = 128u;

@group(0) @binding(0)
var<storage,read_write> arr_i32: array<atomic<i32>, SIZE>;
@group(0) @binding(1)
var<storage,read_write> arr_u32: array<atomic<u32>, SIZE>;

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_i32() {
    for(var i = 0u; i < SIZE; i++) {
        var old = atomicLoad(&arr_i32[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ = bitcast<i32>(bitcast<f32>(old) + 1.0);
            let result = atomicCompareExchangeWeak(&arr_i32[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_u32() {
    for(var i = 0u; i < SIZE; i++) {
        var old = atomicLoad(&arr_u32[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ = bitcast<u32>(bitcast<f32>(old) + 1.0);
            let result = atomicCompareExchangeWeak(&arr_u32[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}
