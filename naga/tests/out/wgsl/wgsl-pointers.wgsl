struct DynamicArray {
    arr: array<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> dynamic_array: DynamicArray;

fn f() {
    var v: mat2x2<f32>;

    let px = (&v[0]);
    (*px) = vec2(10f);
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

@compute @workgroup_size(1, 1, 1) 
fn main() {
    f();
    index_unsized(1i, 1u);
    index_dynamic_array(1i, 1u);
    return;
}
