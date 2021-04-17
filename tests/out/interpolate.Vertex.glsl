#version 400 core
struct FragmentInput {
    vec4 position;
    uint flat1;
    float linear;
    vec2 linear_centroid;
    vec3 linear_sample;
    vec4 perspective;
    float perspective_centroid;
    float perspective_sample;
};

flat out uint _vs2fs_location0;
noperspective out float _vs2fs_location1;
noperspective centroid out vec2 _vs2fs_location2;
noperspective sample out vec3 _vs2fs_location3;
smooth out vec4 _vs2fs_location4;
smooth centroid out float _vs2fs_location5;
smooth sample out float _vs2fs_location6;

void main() {
    FragmentInput out1;
    out1.position = vec4(2.0, 4.0, 5.0, 6.0);
    out1.flat1 = 8u;
    out1.linear = 27.0;
    out1.linear_centroid = vec2(64.0, 125.0);
    out1.linear_sample = vec3(216.0, 343.0, 512.0);
    out1.perspective = vec4(729.0, 1000.0, 1331.0, 1728.0);
    out1.perspective_centroid = 2197.0;
    out1.perspective_sample = 2744.0;
    gl_Position = out1.position;
    _vs2fs_location0 = out1.flat1;
    _vs2fs_location1 = out1.linear;
    _vs2fs_location2 = out1.linear_centroid;
    _vs2fs_location3 = out1.linear_sample;
    _vs2fs_location4 = out1.perspective;
    _vs2fs_location5 = out1.perspective_centroid;
    _vs2fs_location6 = out1.perspective_sample;
    return;
}

