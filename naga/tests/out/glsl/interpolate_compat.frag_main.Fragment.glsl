#version 400 core
struct FragmentInput {
    vec4 position;
    uint _flat;
    uint flat_either;
    float _linear;
    vec2 linear_centroid;
    vec3 linear_sample;
    vec3 linear_center;
    vec4 perspective;
    float perspective_centroid;
    float perspective_sample;
    float perspective_center;
};
flat in uint _vs2fs_location0;
flat in uint _vs2fs_location2;
noperspective in float _vs2fs_location3;
noperspective centroid in vec2 _vs2fs_location4;
noperspective sample in vec3 _vs2fs_location6;
noperspective in vec3 _vs2fs_location7;
smooth in vec4 _vs2fs_location8;
smooth centroid in float _vs2fs_location9;
smooth sample in float _vs2fs_location10;
smooth in float _vs2fs_location11;

void main() {
    FragmentInput val = FragmentInput(gl_FragCoord, _vs2fs_location0, _vs2fs_location2, _vs2fs_location3, _vs2fs_location4, _vs2fs_location6, _vs2fs_location7, _vs2fs_location8, _vs2fs_location9, _vs2fs_location10, _vs2fs_location11);
    return;
}

