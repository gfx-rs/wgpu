#version 400 core
struct FragmentInput {
    vec4 position;
    uint flat1;
    float linear;
    vec2 centroid1;
    vec3 sample1;
    vec4 perspective;
};

out uint _vs2fs_location0;
out float _vs2fs_location1;
out vec2 _vs2fs_location2;
out vec3 _vs2fs_location3;
out vec4 _vs2fs_location4;

void main() {
    FragmentInput out1;
    out1.position = vec4(2.0, 4.0, 5.0, 6.0);
    out1.flat1 = 8u;
    out1.linear = 27.0;
    out1.centroid1 = vec2(64.0, 125.0);
    out1.sample1 = vec3(216.0, 343.0, 512.0);
    out1.perspective = vec4(729.0, 1000.0, 1331.0, 1728.0);
    gl_Position = out1.position;
    _vs2fs_location0 = out1.flat1;
    _vs2fs_location1 = out1.linear;
    _vs2fs_location2 = out1.centroid1;
    _vs2fs_location3 = out1.sample1;
    _vs2fs_location4 = out1.perspective;
    return;
}

