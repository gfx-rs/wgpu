struct type_1 {
    member: u32,
}

var<private> global: u32;
@group(0) @binding(0)
var<storage, read_write> global_1: type_1;

fn function_() {
    let _e3 = global;
    let _e5 = subgroupBallot((_e3 != 0u));
    let unnamed = _e5.x;
    let unnamed_1 = _e5.y;
    let unnamed_2 = _e5.z;
    let unnamed_3 = _e5.w;
    let _e6 = select(select(select(firstTrailingBit(unnamed_3) + 96u, firstTrailingBit(unnamed_2) + 64u, unnamed_2 != 0u), firstTrailingBit(unnamed_1) + 32u, unnamed_1 != 0u), firstTrailingBit(unnamed), unnamed != 0u);
    let unnamed_4 = _e5.x;
    let unnamed_5 = _e5.y;
    let unnamed_6 = _e5.z;
    let unnamed_7 = _e5.w;
    let _e7 = select(select(select(firstLeadingBit(unnamed_4), firstLeadingBit(unnamed_5) + 32u, unnamed_5 != 0u), firstLeadingBit(unnamed_6) + 64u, unnamed_6 != 0u), firstLeadingBit(unnamed_7) + 96u, unnamed_7 != 0u);
    global_1.member = (_e6 + _e7);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(subgroup_invocation_id) param: u32) {
    global = param;
    function_();
}
