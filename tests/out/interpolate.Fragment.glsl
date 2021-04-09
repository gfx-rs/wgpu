#version 400 core
struct FragmentInput {
    vec4 position;
    uint flat1;
    float linear;
    vec2 centroid1;
    vec3 sample1;
    vec4 perspective;
};

flat in uint _vs2fs_location0;
noperspective in float _vs2fs_location1;
centroid in vec2 _vs2fs_location2;
sample in vec3 _vs2fs_location3;
smooth in vec4 _vs2fs_location4;

void main() {
    FragmentInput val = FragmentInput(gl_FragCoord, _vs2fs_location0, _vs2fs_location1, _vs2fs_location2, _vs2fs_location3, _vs2fs_location4);
    return;
}

