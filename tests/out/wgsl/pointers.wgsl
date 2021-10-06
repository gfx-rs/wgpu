[[block]]
struct DynamicArray {
    array1: [[stride(4)]] array<u32>;
};

fn f() {
    var v: vec2<i32>;

    let px: ptr<function, i32> = (&v.x);
    (*px) = 10;
    return;
}

fn index_dynamic_array(p: ptr<workgroup, DynamicArray>, i: i32, v1: u32) -> u32 {
    let old: u32 = (*p).array1[i];
    (*p).array1[i] = v1;
    return old;
}

