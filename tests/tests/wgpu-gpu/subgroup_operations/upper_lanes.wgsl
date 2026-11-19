@group(0)
@binding(0)
var<storage, read_write> storage_buffer: array<vec2<u32>>;

fn add_result_to_mask(mask: ptr<function, vec2<u32>>, index: u32, value: bool) {
   (*mask)[index / 32u] |= u32(value) << (index % 32u);
}

// Every result here depends on the upper half of the subgroup: lanes 32-63 of a 64-wide subgroup.
@compute
@workgroup_size(128)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) id: u32,
) {
    var passed = vec2<u32>(0u);
    let last = subgroup_size - 1u;
    let half = subgroup_size / 2u;

    var upper_half = vec4<u32>(0u);
    for (var i = half; i < subgroup_size; i += 1u) {
        upper_half[i / 32u] |= 1u << (i % 32u);
    }
    add_result_to_mask(&passed, 0u, all(subgroupBallot(id >= half) == upper_half));

    var last_lane = vec4<u32>(0u);
    last_lane[last / 32u] = 1u << (last % 32u);
    add_result_to_mask(&passed, 1u, all(subgroupBallot(id == last) == last_lane));

    add_result_to_mask(&passed, 2u, subgroupShuffle(id, last) == last);
    let down = subgroupShuffleDown(id, half + 1u);
    add_result_to_mask(&passed, 3u, down == id + half + 1u || id + half + 1u > last);
    let up = subgroupShuffleUp(id, half + 1u);
    add_result_to_mask(&passed, 4u, up == id - half - 1u || id < half + 1u);
    add_result_to_mask(&passed, 5u, subgroupShuffleXor(id, half) == (id ^ half));

    add_result_to_mask(&passed, 6u, !subgroupAll(id != last));
    add_result_to_mask(&passed, 7u, subgroupAny(id == last));
    add_result_to_mask(&passed, 8u, subgroupMax(id) == last);
    add_result_to_mask(&passed, 9u, subgroupMin(last - id) == 0u);
    add_result_to_mask(&passed, 10u, subgroupOr(select(0u, 0x80000000u, id == last)) == 0x80000000u);

    let exclusive = subgroupExclusiveAdd(1u);
    let inclusive = subgroupInclusiveAdd(1u);
    add_result_to_mask(&passed, 11u, exclusive == id && inclusive == id + 1u);

    add_result_to_mask(&passed, 12u, quadBroadcast(id, 3u) == (id | 3u));

    // Lane 63 doesn't exist in narrower subgroups, where the value is indeterminate and not checked.
    let lane_63 = subgroupBroadcast(id, 63u);
    add_result_to_mask(&passed, 13u, subgroup_size < 64u || lane_63 == 63u);

    storage_buffer[global_id.x] = passed;
}
