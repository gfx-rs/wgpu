#version 310 es

precision highp float;
precision highp int;

struct OurVertexShaderOutput {
    vec4 position;
    vec2 texcoord;
};
layout(location = 0) in vec2 _p2vs_location0;
layout(location = 0) smooth out vec2 _vs2fs_location0;

void main() {
    vec2 xy = _p2vs_location0;
    OurVertexShaderOutput vsOutput = OurVertexShaderOutput(vec4(0.0), vec2(0.0));
    vsOutput.position = vec4(xy, 0.0, 1.0);
    OurVertexShaderOutput _e6 = vsOutput;
    gl_Position = _e6.position;
    _vs2fs_location0 = _e6.texcoord;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

