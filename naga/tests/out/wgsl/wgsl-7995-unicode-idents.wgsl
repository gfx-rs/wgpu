@group(0) @binding(0) 
var<storage> asdf: f32;

fn compute() -> f32 {
    let _e1 = asdf;
    let u03b8_2_ = (_e1 + 9001f);
    return u03b8_2_;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = compute();
    return;
}
