const SIZE: u32 = 128u;

var<workgroup> arr_i32: array<i32, SIZE>;

@compute @workgroup_size(4)
fn test_workgroupUniformLoad(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let x = &arr_i32[workgroup_id.x];
    let val = workgroupUniformLoad(x);
    if val > 10 {
        workgroupBarrier();
    }
}
