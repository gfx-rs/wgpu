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

    var mask = 1u << 0u;
    passed |= mask * u32(num_subgroups == 128u / subgroup_size);
    mask = 1u << 1u;
    passed |= mask * u32(subgroup_id == global_id.x / subgroup_size);
    mask = 1u << 2u;
    passed |= mask * u32(subgroup_invocation_id == global_id.x % subgroup_size);

    var expected_ballot = vec4<u32>(0u);
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected_ballot[i / 32u] |= ((global_id.x - subgroup_invocation_id + i) & 1u) << (i % 32u);
    }
    mask = 1u << 3u;
    passed |= mask * u32(dot(vec4<u32>(1u), vec4<u32>(subgroupBallot((subgroup_invocation_id & 1u) == 1u) == expected_ballot)) == 4u);

    mask = 1u << 4u;
    passed |= mask * u32(subgroupAll(true));
    mask = 1u << 5u;
    passed |= mask * u32(!subgroupAll(subgroup_invocation_id != 0u));

    mask = 1u << 6u;
    passed |= mask * u32(subgroupAny(subgroup_invocation_id == 0u));
    mask = 1u << 7u;
    passed |= mask * u32(!subgroupAny(false));

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 8u;
    passed |= mask * u32(subgroupAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 9u;
    passed |= mask * u32(subgroupMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = max(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    mask = 1u << 10u;
    passed |= mask * u32(subgroupMax(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = min(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    mask = 1u << 11u;
    passed |= mask * u32(subgroupMin(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected &= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 12u;
    passed |= mask * u32(subgroupAnd(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected |= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 13u;
    passed |= mask * u32(subgroupOr(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected ^= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 14u;
    passed |= mask * u32(subgroupXor(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 15u;
    passed |= mask * u32(subgroupPrefixExclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 16u;
    passed |= mask * u32(subgroupPrefixExclusiveMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 17u;
    passed |= mask * u32(subgroupPrefixInclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    mask = 1u << 18u;
    passed |= mask * u32(subgroupPrefixInclusiveMul(global_id.x + 1u) == expected);

    mask = 1u << 19u;
    passed |= mask * u32(subgroupBroadcastFirst(u32(subgroup_invocation_id != 0u)) == 0u);
    mask = 1u << 20u;
    passed |= mask * u32(subgroupBroadcastFirst(u32(subgroup_invocation_id == 0u)) == 1u);
    mask = 1u << 21u;
    passed |= mask * u32(subgroupBroadcast(subgroup_invocation_id, 1u) == 1u);
    mask = 1u << 22u;
    passed |= mask * u32(subgroupShuffle(subgroup_invocation_id, subgroup_invocation_id) == subgroup_invocation_id);
    mask = 1u << 23u;
    passed |= mask * u32(subgroupShuffle(subgroup_invocation_id, subgroup_size - 1u - subgroup_invocation_id) == subgroup_size - 1u - subgroup_invocation_id);
    mask = 1u << 24u;
    passed |= mask * u32(subgroup_invocation_id == subgroup_size - 1u || subgroupShuffleDown(subgroup_invocation_id, 1u) == subgroup_invocation_id + 1u);
    mask = 1u << 25u;
    passed |= mask * u32(subgroup_invocation_id == 0u || subgroupShuffleUp(subgroup_invocation_id, 1u) == subgroup_invocation_id - 1u);
    mask = 1u << 26u;
    passed |= mask * u32(subgroupShuffleXor(subgroup_invocation_id, subgroup_size - 1u) == (subgroup_invocation_id ^ (subgroup_size - 1u)));

    mask = 1u << 27u;
    if subgroup_invocation_id % 2u == 0u {
        passed |= mask * u32(subgroupAdd(1u) == (subgroup_size / 2u));
    } else {
        passed |= mask * u32(subgroupAdd(1u) == (subgroup_size / 2u));
    }

    mask = 1u << 28u;
    switch subgroup_invocation_id % 3u {
        case 0u: {
            passed |= mask * u32(subgroupBroadcastFirst(subgroup_invocation_id) == 0u);
        }
        case 1u: {
            passed |= mask * u32(subgroupBroadcastFirst(subgroup_invocation_id) == 1u);
        }
        case 2u: {
            passed |= mask * u32(subgroupBroadcastFirst(subgroup_invocation_id) == 2u);
        }
        default {  }
    }

    mask = 1u << 29u;
    expected = 0u;
    for (var i = subgroup_size; i >= 0u; i -= 1u) {
        expected = subgroupAdd(1u);
        if i == subgroup_invocation_id {
            break;
        }
    }
    passed |= mask * u32(expected == (subgroup_invocation_id + 1u));

    // Keep this test last, verify we are still convergent after running other tests
    mask = 1u << 30u;
    passed |= mask * u32(subgroup_size == subgroupAdd(1u));

    // Increment TEST_COUNT in subgroup_operations/mod.rs if adding more tests

    storage_buffer[global_id.x] = passed;
}
