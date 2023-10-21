@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(num_subgroups) num_subgroups: u32, @builtin(subgroup_id) subgroup_id: u32, @builtin(subgroup_size) subgroup_size: u32, @builtin(subgroup_invocation_id) subgroup_invocation_id: u32) {
    subgroupBarrier();
    let _e8 = subgroupBallot(((subgroup_invocation_id & 1u) == 1u));
    let _e9 = subgroupBallot();
    let _e12 = subgroupAll((subgroup_invocation_id != 0u));
    let _e15 = subgroupAny((subgroup_invocation_id == 0u));
    let _e16 = subgroupAdd(subgroup_invocation_id);
    let _e17 = subgroupMul(subgroup_invocation_id);
    let _e18 = subgroupMin(subgroup_invocation_id);
    let _e19 = subgroupMax(subgroup_invocation_id);
    let _e20 = subgroupAnd(subgroup_invocation_id);
    let _e21 = subgroupOr(subgroup_invocation_id);
    let _e22 = subgroupXor(subgroup_invocation_id);
    let _e23 = subgroupPrefixExclusiveAdd(subgroup_invocation_id);
    let _e24 = subgroupPrefixExclusiveMul(subgroup_invocation_id);
    let _e25 = subgroupPrefixInclusiveAdd(subgroup_invocation_id);
    let _e26 = subgroupPrefixInclusiveMul(subgroup_invocation_id);
    let _e27 = subgroupBroadcastFirst(subgroup_invocation_id);
    let _e29 = subgroupBroadcast(subgroup_invocation_id, 4u);
    let _e33 = subgroupShuffle(subgroup_invocation_id, ((subgroup_size - 1u) - subgroup_invocation_id));
    let _e35 = subgroupShuffleDown(subgroup_invocation_id, 1u);
    let _e37 = subgroupShuffleUp(subgroup_invocation_id, 1u);
    let _e40 = subgroupShuffleXor(subgroup_invocation_id, (subgroup_size - 1u));
    return;
}
