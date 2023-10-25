struct Ah {
    inner: array<f32, 2>,
};
@group(0) @binding(0)
var<storage> ah: Ah;

@compute @workgroup_size(1)
fn cs_main() {
    let ah = ah;
}
