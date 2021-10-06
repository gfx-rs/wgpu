fn f() {
   var v: vec2<i32>;
   let px = &v.x;
   *px = 10;
}

[[block]]
struct DynamicArray {
    array: array<u32>;
};

fn index_dynamic_array(p: ptr<workgroup, DynamicArray>, i: i32, v: u32) -> u32 {
   let old = (*p).array[i];
   (*p).array[i] = v;
   return old;
}
