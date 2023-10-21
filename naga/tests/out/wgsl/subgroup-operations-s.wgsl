var<private> num_subgroups_1: u32;
var<private> subgroup_id_1: u32;
var<private> subgroup_size_1: u32;
var<private> subgroup_invocation_id_1: u32;

fn main_1() {
    let _e5 = subgroup_size_1;
    let _e6 = subgroup_invocation_id_1;
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
    let _e22 = subgroupPrefixExclusiveAdd(_e6);
    let _e23 = subgroupPrefixExclusiveMul(_e6);
    let _e24 = subgroupPrefixInclusiveAdd(_e6);
    let _e25 = subgroupPrefixInclusiveMul(_e6);
    let _e26 = subgroupBroadcastFirst(_e6);
    let _e27 = subgroupBroadcast(_e6, 4u);
    let _e30 = subgroupShuffle(_e6, ((_e5 - 1u) - _e6));
    let _e31 = subgroupShuffleDown(_e6, 1u);
    let _e32 = subgroupShuffleUp(_e6, 1u);
    let _e34 = subgroupShuffleXor(_e6, (_e5 - 1u));
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(num_subgroups) num_subgroups: u32, @builtin(subgroup_id) subgroup_id: u32, @builtin(subgroup_size) subgroup_size: u32, @builtin(subgroup_invocation_id) subgroup_invocation_id: u32) {
    num_subgroups_1 = num_subgroups;
    subgroup_id_1 = subgroup_id;
    subgroup_size_1 = subgroup_size;
    subgroup_invocation_id_1 = subgroup_invocation_id;
    main_1();
}
