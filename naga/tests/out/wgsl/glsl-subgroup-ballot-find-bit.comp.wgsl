struct Output {
    result: u32,
}

@group(0) @binding(0)
var<storage, read_write> global: Output;
var<private> gl_SubgroupInvocationID_1: u32;
var<private> gl_SubgroupSize_1: u32;

fn main_1() {
    var mask: vec4<u32>;
    var lsb: u32;
    var msb: u32;

    let _e4 = gl_SubgroupInvocationID_1;
    let _e5 = gl_SubgroupSize_1;
    mask = vec4<u32>(_e4, _e5, 3u, 4u);
    let _e10 = mask;
    let unnamed = _e10.x;
    let unnamed_1 = _e10.y;
    let unnamed_2 = _e10.z;
    let unnamed_3 = _e10.w;
    let _e11 = select(select(select(firstTrailingBit(unnamed_3) + 96u, firstTrailingBit(unnamed_2) + 64u, unnamed_2 != 0u), firstTrailingBit(unnamed_1) + 32u, unnamed_1 != 0u), firstTrailingBit(unnamed), unnamed != 0u);
    lsb = _e11;
    let _e13 = mask;
    let unnamed_4 = _e13.x;
    let unnamed_5 = _e13.y;
    let unnamed_6 = _e13.z;
    let unnamed_7 = _e13.w;
    let _e14 = select(select(select(firstLeadingBit(unnamed_4), firstLeadingBit(unnamed_5) + 32u, unnamed_5 != 0u), firstLeadingBit(unnamed_6) + 64u, unnamed_6 != 0u), firstLeadingBit(unnamed_7) + 96u, unnamed_7 != 0u);
    msb = _e14;
    let _e16 = lsb;
    let _e17 = msb;
    global.result = (_e16 + _e17);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(subgroup_invocation_id) gl_SubgroupInvocationID: u32, @builtin(subgroup_size) gl_SubgroupSize: u32) {
    gl_SubgroupInvocationID_1 = gl_SubgroupInvocationID;
    gl_SubgroupSize_1 = gl_SubgroupSize;
    main_1();
    return;
}
