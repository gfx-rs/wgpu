@compute @workgroup_size(1)
fn main(
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    subgroupBarrier();

    subgroupBallot((subgroup_invocation_id & 1u) == 1u);
    subgroupBallot();

    subgroupAll(subgroup_invocation_id != 0u);
    subgroupAny(subgroup_invocation_id == 0u);
    subgroupAdd(subgroup_invocation_id);
    subgroupMul(subgroup_invocation_id);
    subgroupMin(subgroup_invocation_id);
    subgroupMax(subgroup_invocation_id);
    subgroupAnd(subgroup_invocation_id);
    subgroupOr(subgroup_invocation_id);
    subgroupXor(subgroup_invocation_id);
    subgroupPrefixExclusiveAdd(subgroup_invocation_id);
    subgroupPrefixExclusiveMul(subgroup_invocation_id);
    subgroupPrefixInclusiveAdd(subgroup_invocation_id);
    subgroupPrefixInclusiveMul(subgroup_invocation_id);

    subgroupBroadcastFirst(subgroup_invocation_id);
    subgroupBroadcast(subgroup_invocation_id, 4u);
    subgroupShuffle(subgroup_invocation_id, subgroup_size - 1u - subgroup_invocation_id);
    subgroupShuffleDown(subgroup_invocation_id, 1u);
    subgroupShuffleUp(subgroup_invocation_id, 1u);
    subgroupShuffleXor(subgroup_invocation_id, subgroup_size - 1u);
}
