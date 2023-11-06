@group(0)
@binding(0)
var<storage, read_write> storage_buffer: array<u32>;

@compute
@workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    var passed = 0u;
    var expected: u32;

    passed += u32(num_subgroups == 128u / subgroup_size);
    passed += u32(subgroup_id == global_id.x / subgroup_size);
    passed += u32(subgroup_invocation_id == global_id.x % subgroup_size);

    var expected_ballot = vec4<u32>(0u);
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected_ballot[i / 32u] |= ((global_id.x - subgroup_invocation_id + i) & 1u) << (i % 32u);
    }
    passed += u32(dot(vec4<u32>(1u), vec4<u32>(subgroupBallot((subgroup_invocation_id & 1u) == 1u) == expected_ballot)) == 4u);

    passed += u32(subgroupAll(true));
    passed += u32(!subgroupAll(subgroup_invocation_id != 0u));

    passed += u32(subgroupAny(subgroup_invocation_id == 0u));
    passed += u32(!subgroupAny(false));

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = max(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    passed += u32(subgroupMax(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = min(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    passed += u32(subgroupMin(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected &= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupAnd(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected |= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupOr(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected ^= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupXor(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupPrefixExclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupPrefixExclusiveMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupPrefixInclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    passed += u32(subgroupPrefixInclusiveMul(global_id.x + 1u) == expected);

    passed += u32(subgroupBroadcastFirst(u32(subgroup_invocation_id != 0u)) == 0u);
    passed += u32(subgroupBroadcastFirst(u32(subgroup_invocation_id == 0u)) == 1u);
    passed += u32(subgroupBroadcast(subgroup_invocation_id, 1u) == 1u);
    passed += u32(subgroupShuffle(subgroup_invocation_id, subgroup_invocation_id) == subgroup_invocation_id);
    passed += u32(subgroupShuffle(subgroup_invocation_id, subgroup_size - 1u - subgroup_invocation_id) == subgroup_size - 1u - subgroup_invocation_id);
    passed += u32(subgroup_invocation_id == subgroup_size - 1u || subgroupShuffleDown(subgroup_invocation_id, 1u) == subgroup_invocation_id + 1u);
    passed += u32(subgroup_invocation_id == 0u || subgroupShuffleUp(subgroup_invocation_id, 1u) == subgroup_invocation_id - 1u);
    passed += u32(subgroupShuffleXor(subgroup_invocation_id, subgroup_size - 1u) == (subgroup_invocation_id ^ (subgroup_size - 1u)));

    storage_buffer[global_id.x] = passed;
}
