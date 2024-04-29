@group(0)
@binding(0)
var<storage, read_write> storage_buffer: array<u32>;

var<workgroup> workgroup_buffer: u32;

fn add_result_to_mask(mask: ptr<function, u32>, index: u32, value: bool) {
   (*mask) |= u32(value) << index;
}

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

    add_result_to_mask(&passed, 0u, num_subgroups == 128u / subgroup_size);
    add_result_to_mask(&passed, 1u, subgroup_id == global_id.x / subgroup_size);
    add_result_to_mask(&passed, 2u, subgroup_invocation_id == global_id.x % subgroup_size);

    var expected_ballot = vec4<u32>(0u);
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected_ballot[i / 32u] |= ((global_id.x - subgroup_invocation_id + i) & 1u) << (i % 32u);
    }
    add_result_to_mask(&passed, 3u, dot(vec4<u32>(1u), vec4<u32>(subgroupBallot((subgroup_invocation_id & 1u) == 1u) == expected_ballot)) == 4u);

    add_result_to_mask(&passed, 4u, subgroupAll(true));
    add_result_to_mask(&passed, 5u, !subgroupAll(subgroup_invocation_id != 0u));

    add_result_to_mask(&passed, 6u, subgroupAny(subgroup_invocation_id == 0u));
    add_result_to_mask(&passed, 7u, !subgroupAny(false));

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 8u, subgroupAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 9u, subgroupMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = max(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    add_result_to_mask(&passed, 10u, subgroupMax(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected = min(expected, global_id.x - subgroup_invocation_id + i + 1u);
    }
    add_result_to_mask(&passed, 11u, subgroupMin(global_id.x + 1u) == expected);

    expected = 0xFFFFFFFFu;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected &= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 12u, subgroupAnd(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected |= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 13u, subgroupOr(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_size; i += 1u) {
        expected ^= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 14u, subgroupXor(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 15u, subgroupExclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i < subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 16u, subgroupExclusiveMul(global_id.x + 1u) == expected);

    expected = 0u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected += global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 17u, subgroupInclusiveAdd(global_id.x + 1u) == expected);

    expected = 1u;
    for(var i = 0u; i <= subgroup_invocation_id; i += 1u) {
        expected *= global_id.x - subgroup_invocation_id + i + 1u;
    }
    add_result_to_mask(&passed, 18u, subgroupInclusiveMul(global_id.x + 1u) == expected);

    add_result_to_mask(&passed, 19u, subgroupBroadcastFirst(u32(subgroup_invocation_id != 0u)) == 0u);
    add_result_to_mask(&passed, 20u, subgroupBroadcastFirst(u32(subgroup_invocation_id == 0u)) == 1u);
    add_result_to_mask(&passed, 21u, subgroupBroadcast(subgroup_invocation_id, 1u) == 1u);
    add_result_to_mask(&passed, 22u, subgroupShuffle(subgroup_invocation_id, subgroup_invocation_id) == subgroup_invocation_id);
    add_result_to_mask(&passed, 23u, subgroupShuffle(subgroup_invocation_id, subgroup_size - 1u - subgroup_invocation_id) == subgroup_size - 1u - subgroup_invocation_id);
    add_result_to_mask(&passed, 24u, subgroup_invocation_id == subgroup_size - 1u || subgroupShuffleDown(subgroup_invocation_id, 1u) == subgroup_invocation_id + 1u);
    add_result_to_mask(&passed, 25u, subgroup_invocation_id == 0u || subgroupShuffleUp(subgroup_invocation_id, 1u) == subgroup_invocation_id - 1u);
    add_result_to_mask(&passed, 26u, subgroupShuffleXor(subgroup_invocation_id, subgroup_size - 1u) == (subgroup_invocation_id ^ (subgroup_size - 1u)));

    // Mac/Apple will fail this test.
    var passed_27 = false;
    if subgroup_invocation_id % 2u == 0u {
        passed_27 |= subgroupAdd(1u) == (subgroup_size / 2u);
    } else {
        passed_27 |= subgroupAdd(1u) == (subgroup_size / 2u);
    }
    add_result_to_mask(&passed, 27u, passed_27);

    // Mac/Apple will fail this test.
    var passed_28 = false;
    switch subgroup_invocation_id % 3u {
        case 0u: {
            passed_28 = subgroupBroadcastFirst(subgroup_invocation_id) == 0u;
        }
        case 1u: {
            passed_28 = subgroupBroadcastFirst(subgroup_invocation_id) == 1u;
        }
        case 2u: {
            passed_28 = subgroupBroadcastFirst(subgroup_invocation_id) == 2u;
        }
        default {  }
    }
    add_result_to_mask(&passed, 28u, passed_28);

    // Mac/Apple will sometimes fail this test. MacOS 14.3 passes it, so the bug in the metal compiler seems to be fixed.
    expected = 0u;
    for (var i = subgroup_size; i >= 0u; i -= 1u) {
        expected = subgroupAdd(1u);
        if i == subgroup_invocation_id {
            break;
        }
    }
    add_result_to_mask(&passed, 29u, expected == (subgroup_invocation_id + 1u));

    if global_id.x == 0u {
        workgroup_buffer = subgroup_size;
    }
    workgroupBarrier();
    add_result_to_mask(&passed, 30u, workgroup_buffer == subgroup_size);

    // Keep this test last, verify we are still convergent after running other tests
    add_result_to_mask(&passed, 31u, subgroupAdd(1u) == subgroup_size);

    // Increment TEST_COUNT in subgroup_operations/mod.rs if adding more tests

    storage_buffer[global_id.x] = passed;
}
