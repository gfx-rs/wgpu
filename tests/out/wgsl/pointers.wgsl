[[block]]
struct DynamicArray {
    array_: [[stride(4)]] array<u32>;
};

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

