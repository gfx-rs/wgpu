struct Output {
    result: u32,
}

@group(0) @binding(0)
var<storage, read_write> global: Output;
var<private> gl_GlobalInvocationID_1: vec3<u32>;

fn main_1() {
    var mask: vec4<u32>;
    var lsb: u32;
    var msb: u32;

    let _e3 = gl_GlobalInvocationID_1;
    mask = vec4<u32>(_e3.x, 2u, 3u, 4u);
    let _e10 = mask;
    let _e11 = select(select(select(firstTrailingBit(_e10.w) + 96u, firstTrailingBit(_e10.z) + 64u, _e10.z != 0u), firstTrailingBit(_e10.y) + 32u, _e10.y != 0u), firstTrailingBit(_e10.x), _e10.x != 0u);
    lsb = _e11;
    let _e13 = mask;
    let _e14 = select(select(select(firstLeadingBit(_e13.x), firstLeadingBit(_e13.y) + 32u, _e13.y != 0u), firstLeadingBit(_e13.z) + 64u, _e13.z != 0u), firstLeadingBit(_e13.w) + 96u, _e13.w != 0u);
    msb = _e14;
    let _e16 = lsb;
    let _e17 = msb;
    global.result = (_e16 + _e17);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>) {
    gl_GlobalInvocationID_1 = gl_GlobalInvocationID;
    main_1();
    return;
}
