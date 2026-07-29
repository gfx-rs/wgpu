struct type_1 {
    member: u32,
}

var<private> global: u32;
@group(0) @binding(0)
var<storage, read_write> global_1: type_1;

fn function_() {
    let _e3 = global;
    let _e5 = subgroupBallot((_e3 != 0u));
    let _e6 = select(select(select(firstTrailingBit(_e5.w) + 96u, firstTrailingBit(_e5.z) + 64u, _e5.z != 0u), firstTrailingBit(_e5.y) + 32u, _e5.y != 0u), firstTrailingBit(_e5.x), _e5.x != 0u);
    let _e7 = select(select(select(firstLeadingBit(_e5.x), firstLeadingBit(_e5.y) + 32u, _e5.y != 0u), firstLeadingBit(_e5.z) + 64u, _e5.z != 0u), firstLeadingBit(_e5.w) + 96u, _e5.w != 0u);
    global_1.member = (_e6 + _e7);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(subgroup_invocation_id) param: u32) {
    global = param;
    function_();
}
