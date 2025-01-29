const SIZE: u32 = 128u;

var<workgroup> arr_i32_: array<i32, 128>;

@compute @workgroup_size(4, 1, 1) 
fn test_workgroupUniformLoad(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let x = (&arr_i32_[workgroup_id.x]);
    let _e4 = workgroupUniformLoad(x);
    if (_e4 > 10i) {
        workgroupBarrier();
        return;
    } else {
        return;
    }
}
