fn f() {
   var v: vec2<i32>;
   let px = &v.x;
   *px = 10;
}

struct DynamicArray {
    arr: array<u32>
}

@group(0) @binding(0)
var<storage, read_write> dynamic_array: DynamicArray;

fn index_unsized(i: i32, v: u32) {
   let p: ptr<storage, DynamicArray, read_write> = &dynamic_array;

   let val = (*p).arr[i];
   (*p).arr[i] = val + v;
}

fn index_dynamic_array(i: i32, v: u32) {
   let p: ptr<storage, array<u32>, read_write> = &dynamic_array.arr;

   let val = (*p)[i];
   (*p)[i] = val + v;
}
