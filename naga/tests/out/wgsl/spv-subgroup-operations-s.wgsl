var<private> global: u32;
var<private> global_1: u32;
var<private> global_2: u32;
var<private> global_3: u32;

fn function() {
    let _e5 = global_2;
    let _e6 = global_3;
    let _e9 = subgroupBallot(((_e6 & 1u) == 1u));
    let _e10 = subgroupBallot();
    let _e12 = subgroupAll((_e6 != 0u));
    let _e14 = subgroupAny((_e6 == 0u));
    let _e15 = subgroupAdd(_e6);
    let _e16 = subgroupMul(_e6);
    let _e17 = subgroupMin(_e6);
    let _e18 = subgroupMax(_e6);
    let _e19 = subgroupAnd(_e6);
    let _e20 = subgroupOr(_e6);
    let _e21 = subgroupXor(_e6);
    let _e22 = subgroupExclusiveAdd(_e6);
    let _e23 = subgroupExclusiveMul(_e6);
    let _e24 = subgroupInclusiveAdd(_e6);
    let _e25 = subgroupInclusiveMul(_e6);
    let _e26 = subgroupBroadcastFirst(_e6);
    let _e27 = subgroupBroadcast(_e6, 4u);
    let _e30 = subgroupShuffle(_e6, ((_e5 - 1u) - _e6));
    let _e31 = subgroupShuffleDown(_e6, 1u);
    let _e32 = subgroupShuffleUp(_e6, 1u);
    let _e34 = subgroupShuffleXor(_e6, (_e5 - 1u));
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(num_subgroups) param: u32, @builtin(subgroup_id) param_1: u32, @builtin(subgroup_size) param_2: u32, @builtin(subgroup_invocation_id) param_3: u32) {
    global = param;
    global_1 = param_1;
    global_2 = param_2;
    global_3 = param_3;
    function();
}
