#version 400 core
struct FragmentInput {
    vec4 position;
    uint _flat;
    uint flat_first;
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
flat out uint _vs2fs_location0;
flat out uint _vs2fs_location1;
flat out uint _vs2fs_location2;
noperspective out float _vs2fs_location3;
noperspective centroid out vec2 _vs2fs_location4;
noperspective sample out vec3 _vs2fs_location6;
noperspective out vec3 _vs2fs_location7;
smooth out vec4 _vs2fs_location8;
smooth centroid out float _vs2fs_location9;
smooth sample out float _vs2fs_location10;
smooth out float _vs2fs_location11;

void main() {
    FragmentInput out_ = FragmentInput(vec4(0.0), 0u, 0u, 0u, 0.0, vec2(0.0), vec3(0.0), vec3(0.0), vec4(0.0), 0.0, 0.0, 0.0);
    out_.position = vec4(2.0, 4.0, 5.0, 6.0);
    out_._flat = 8u;
    out_.flat_first = 9u;
    out_.flat_either = 10u;
    out_._linear = 27.0;
    out_.linear_centroid = vec2(64.0, 125.0);
    out_.linear_sample = vec3(216.0, 343.0, 512.0);
    out_.linear_center = vec3(255.0, 511.0, 1024.0);
    out_.perspective = vec4(729.0, 1000.0, 1331.0, 1728.0);
    out_.perspective_centroid = 2197.0;
    out_.perspective_sample = 2744.0;
    out_.perspective_center = 2812.0;
    FragmentInput _e41 = out_;
    gl_Position = _e41.position;
    _vs2fs_location0 = _e41._flat;
    _vs2fs_location1 = _e41.flat_first;
    _vs2fs_location2 = _e41.flat_either;
    _vs2fs_location3 = _e41._linear;
    _vs2fs_location4 = _e41.linear_centroid;
    _vs2fs_location6 = _e41.linear_sample;
    _vs2fs_location7 = _e41.linear_center;
    _vs2fs_location8 = _e41.perspective;
    _vs2fs_location9 = _e41.perspective_centroid;
    _vs2fs_location10 = _e41.perspective_sample;
    _vs2fs_location11 = _e41.perspective_center;
    return;
}

