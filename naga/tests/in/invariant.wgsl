@vertex
fn vs() -> @builtin(position) @invariant vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs(@builtin(position) @invariant position: vec4<f32>) { }
