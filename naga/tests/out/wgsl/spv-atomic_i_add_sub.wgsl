struct type_2 {
    member: array<u32>,
}

struct type_4 {
    member: atomic<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_4;
@group(0) @binding(1) 
var<storage, read_write> global_1: type_2;

fn function() {
    let _e6 = atomicAdd((&global.member), 2u);
    let _e7 = atomicSub((&global.member), _e6);
    if (_e6 < arrayLength((&global_1.member))) {
        global_1.member[_e6] = _e7;
    }
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn stagetest_atomic_i_add_sub() {
    function();
}
