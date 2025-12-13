struct OurVertexShaderOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@vertex 
fn vs(@location(0) xy: vec2<f32>) -> OurVertexShaderOutput {
    var vsOutput: OurVertexShaderOutput;

    vsOutput.position = vec4<f32>(xy, 0f, 1f);
    let _e6 = vsOutput;
    return _e6;
}

@fragment 
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1f, 0f, 0f, 1f);
}
