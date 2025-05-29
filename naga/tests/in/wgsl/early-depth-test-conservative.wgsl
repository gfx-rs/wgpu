@fragment
@early_depth_test(less_equal)
fn main(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    return pos.z - 0.1;
}