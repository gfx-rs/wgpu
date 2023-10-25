// only available in the fragment stage
fn derivatives() {
    let x = dpdx(0.0);
    let y = dpdy(0.0);
    let width = fwidth(0.0);
}

// only available in the compute stage
fn barriers() {
    storageBarrier();
    workgroupBarrier();
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    derivatives();
    return vec4<f32>();
}

@compute @workgroup_size(1)
fn compute() {
    barriers();
}