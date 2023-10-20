@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(num_subgroups) num_subgroups: u32, @builtin(subgroup_id) subgroup_id: u32, @builtin(subgroup_size) subgroup_size: u32, @builtin(subgroup_invocation_id) subgroup_invocation_id: u32) {
    subgroupBarrier();
    let _e8 = subgroupBallot(
((subgroup_invocation_id & 1u) == 1u));
    let _e11 = subgroupAll((subgroup_invocation_id != 0u));
    let _e14 = subgroupAny((subgroup_invocation_id == 0u));
    let _e15 = subgroupAdd(subgroup_invocation_id);
    let _e16 = subgroupMul(subgroup_invocation_id);
    let _e17 = subgroupMin(subgroup_invocation_id);
    let _e18 = subgroupMax(subgroup_invocation_id);
    let _e19 = subgroupAnd(subgroup_invocation_id);
    let _e20 = subgroupOr(subgroup_invocation_id);
    let _e21 = subgroupXor(subgroup_invocation_id);
    let _e22 = subgroupPrefixExclusiveAdd(subgroup_invocation_id);
    let _e23 = subgroupPrefixExclusiveMul(subgroup_invocation_id);
    let _e24 = subgroupPrefixInclusiveAdd(subgroup_invocation_id);
    let _e25 = subgroupPrefixInclusiveMul(subgroup_invocation_id);
    let _e26 = subgroupBroadcastFirst(subgroup_invocation_id);
    let _e28 = subgroupBroadcast(subgroup_invocation_id, 4u);
    let _e32 = subgroupShuffle(subgroup_invocation_id, ((subgroup_size - 1u) - subgroup_invocation_id));
    let _e34 = subgroupShuffleDown(subgroup_invocation_id, 1u);
    let _e36 = subgroupShuffleUp(subgroup_invocation_id, 1u);
    let _e39 = subgroupShuffleXor(subgroup_invocation_id, (subgroup_size - 1u));
    return;
}
