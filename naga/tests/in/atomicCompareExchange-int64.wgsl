const SIZE: u32 = 128u;

@group(0) @binding(0)
var<storage,read_write> arr_i64: array<atomic<i64>, SIZE>;
@group(0) @binding(1)
var<storage,read_write> arr_u64: array<atomic<u64>, SIZE>;

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_i64() {
    for(var i = 0u; i < SIZE; i++) {
        var old : i64 = atomicLoad(&arr_i64[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ : i64 = bitcast<i64>(old + 10li);
            let result = atomicCompareExchangeWeak(&arr_i64[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}

@compute @workgroup_size(1)
fn test_atomic_compare_exchange_u64() {
    for(var i = 0u; i < SIZE; i++) {
        var old : u64 = atomicLoad(&arr_u64[i]);
        var exchanged = false;
        while(!exchanged) {
            let new_ : u64 = bitcast<u64>(old + 10lu);
            let result = atomicCompareExchangeWeak(&arr_u64[i], old, new_);
            old = result.old_value;
            exchanged = result.exchanged;
        }
    }
}
