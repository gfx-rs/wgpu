requires fragment_depth;

@fragment
fn greater() -> @builtin(frag_depth, greater) f32 {
    return 0.5;
}

@fragment
fn less() -> @builtin(frag_depth, less,) f32 {
    return 0.5;
}

@fragment
fn plain() -> @builtin(frag_depth,) f32 {
    return 0.5;
}

struct StructDepthOutput {
    @builtin(frag_depth, greater) depth: f32,
}

@fragment
fn struct_greater() -> StructDepthOutput {
    return StructDepthOutput(0.5);
}
