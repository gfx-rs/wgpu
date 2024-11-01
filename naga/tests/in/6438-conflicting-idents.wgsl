struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex fn vs(
    @location(0) xy: vec2f
) -> OurVertexShaderOutput {
    var vsOutput: OurVertexShaderOutput;
    vsOutput.position = vec4f(xy, 0.0, 1.0);
    return vsOutput;
}

@fragment fn fs() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
