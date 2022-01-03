#version 400 core
struct FragmentInput {
    vec4 position;
    uint flat_;
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
    FragmentInput out_ = FragmentInput(vec4(0.0, 0.0, 0.0, 0.0), 0u, 0.0, vec2(0.0, 0.0), vec3(0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0), 0.0, 0.0);
    out_.position = vec4(2.0, 4.0, 5.0, 6.0);
    out_.flat_ = 8u;
    out_.linear = 27.0;
    out_.linear_centroid = vec2(64.0, 125.0);
    out_.linear_sample = vec3(216.0, 343.0, 512.0);
    out_.perspective = vec4(729.0, 1000.0, 1331.0, 1728.0);
    out_.perspective_centroid = 2197.0;
    out_.perspective_sample = 2744.0;
    FragmentInput _e30 = out_;
    gl_Position = _e30.position;
    _vs2fs_location0 = _e30.flat_;
    _vs2fs_location1 = _e30.linear;
    _vs2fs_location2 = _e30.linear_centroid;
    _vs2fs_location3 = _e30.linear_sample;
    _vs2fs_location4 = _e30.perspective;
    _vs2fs_location5 = _e30.perspective_centroid;
    _vs2fs_location6 = _e30.perspective_sample;
    return;
}

