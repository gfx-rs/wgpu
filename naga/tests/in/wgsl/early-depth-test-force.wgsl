@fragment
@early_depth_test(force)
fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.4, 0.3, 0.2, 0.1);
}