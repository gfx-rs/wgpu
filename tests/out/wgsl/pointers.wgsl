[[block]]
struct DynamicArray {
    array_: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> dynamic_array: DynamicArray;

fn f() {
    var v: vec2<i32>;

    let px: ptr<function, i32> = (&v.x);
    (*px) = 10;
    return;
}

fn index_dynamic_array(p: ptr<workgroup, DynamicArray>, i: i32, v_1: u32) -> u32 {
    let old: u32 = (*p).array_[i];
    (*p).array_[i] = v_1;
    return old;
}

fn index_unsized(i_1: i32, v_2: u32) {
    let val: u32 = dynamic_array.array_[i_1];
    dynamic_array.array_[i_1] = (val + v_2);
    return;
}

fn index_dynamic_array_1(i_2: i32, v_3: u32) {
    let p_1: ptr<storage, array<u32>, read_write> = (&dynamic_array.array_);
    let val_1: u32 = (*p_1)[i_2];
    (*p_1)[i_2] = (val_1 + v_3);
    return;
}

