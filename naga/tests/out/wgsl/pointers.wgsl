struct DynamicArray {
    arr: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> dynamic_array: DynamicArray;

fn f() {
    var v: vec2<i32>;

    let px = (&v.x);
    (*px) = 10i;
    return;
}

fn index_unsized(i: i32, v_1: u32) {
    let val = dynamic_array.arr[i];
    dynamic_array.arr[i] = (val + v_1);
    return;
}

fn index_dynamic_array(i_1: i32, v_2: u32) {
    let p = (&dynamic_array.arr);
    let val_1 = (*p)[i_1];
    (*p)[i_1] = (val_1 + v_2);
    return;
}

