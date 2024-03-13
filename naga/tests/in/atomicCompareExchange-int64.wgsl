const SIZE: u64 = 128u;

@group(0) @binding(0)
var<storage,read_write> arr_i64: array<atomic<i64>, SIZE>;
@group(0) @binding(1)
var<storage,read_write> arr_u64: array<atomic<u64>, SIZE>;

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_i64() {
    for(var i = 0u; i < SIZE; i++) {
        var old = atomicLoad(&arr_i64[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ = bitcast<i64>(bitcast<f32>(old) + 1.0);
            let result = atomicCompareExchangeWeak(&arr_i64[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_u64() {
    for(var i = 0u; i < SIZE; i++) {
        var old = atomicLoad(&arr_u64[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ = bitcast<u64>(bitcast<f32>(old) + 1.0);
            let result = atomicCompareExchangeWeak(&arr_u64[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}
