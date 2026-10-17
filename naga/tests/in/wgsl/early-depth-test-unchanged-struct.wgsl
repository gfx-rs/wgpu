struct ColorAndDepth {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

struct OnlyDepth {
    @builtin(frag_depth) depth: f32,
}

@fragment
@early_depth_test(unchanged)
fn color_and_depth(@builtin(position) pos: vec4<f32>) -> ColorAndDepth {
    return ColorAndDepth(vec4<f32>(0.4, 0.3, 0.2, 0.1), pos.z);
}

@fragment
@early_depth_test(unchanged)
fn only_depth(@builtin(position) pos: vec4<f32>) -> OnlyDepth {
    return OnlyDepth(pos.z);
}

@fragment
fn unrestricted(@builtin(position) pos: vec4<f32>) -> ColorAndDepth {
    return ColorAndDepth(vec4<f32>(0.1, 0.2, 0.3, 0.4), pos.z * 2.0);
}
