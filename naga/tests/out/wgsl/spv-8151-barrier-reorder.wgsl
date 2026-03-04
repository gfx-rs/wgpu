struct type_3 {
    member: array<u32>,
}

var<private> global: vec3<u32>;
@group(0) @binding(0) 
var<storage, read_write> global_1: type_3;
var<workgroup> global_2: u32;

fn function_() {
    let _e6 = global;
    let _e8 = (_e6.x == 0u);
    if _e8 {
        global_2 = 1u;
    }
    workgroupBarrier();
    let _e9 = global_2;
    workgroupBarrier();
    global_1.member[_e6.x] = _e9;
    if _e8 {
        global_2 = 2u;
    }
    return;
}

@compute @workgroup_size(2, 1, 1) 
fn barrier_reorder_bug(@builtin(local_invocation_id) param: vec3<u32>) {
    global = param;
    function_();
}
