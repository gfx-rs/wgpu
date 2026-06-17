/////////////////////////////////////
// Entry point: "vert_main" (vert) //
/////////////////////////////////////
#version 460

struct FragmentInput
{
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

layout(location = 0) flat out uint _flat;
layout(location = 2) flat out uint flat_either;
layout(location = 3) noperspective out float _linear;
layout(location = 4) noperspective centroid out vec2 linear_centroid;
layout(location = 6) noperspective sample out vec3 linear_sample;
layout(location = 7) noperspective out vec3 linear_center;
layout(location = 8) out vec4 perspective;
layout(location = 9) centroid out float perspective_centroid;
layout(location = 10) sample out float perspective_sample;
layout(location = 11) out float perspective_center;

void main()
{
    FragmentInput _out = FragmentInput(vec4(0.0), 0u, 0u, 0.0, vec2(0.0), vec3(0.0), vec3(0.0), vec4(0.0), 0.0, 0.0, 0.0);
    gl_PointSize = 1.0;
    _out.position = vec4(2.0, 4.0, 5.0, 6.0);
    _out._flat = 8u;
    _out.flat_either = 10u;
    _out._linear = 27.0;
    _out.linear_centroid = vec2(64.0, 125.0);
    _out.linear_sample = vec3(216.0, 343.0, 512.0);
    _out.linear_center = vec3(255.0, 511.0, 1024.0);
    _out.perspective = vec4(729.0, 1000.0, 1331.0, 1728.0);
    _out.perspective_centroid = 2197.0;
    _out.perspective_sample = 2744.0;
    _out.perspective_center = 2812.0;
    gl_Position = _out.position;
    gl_Position.y = -gl_Position.y;
    _flat = _out._flat;
    flat_either = _out.flat_either;
    _linear = _out._linear;
    linear_centroid = _out.linear_centroid;
    linear_sample = _out.linear_sample;
    linear_center = _out.linear_center;
    perspective = _out.perspective;
    perspective_centroid = _out.perspective_centroid;
    perspective_sample = _out.perspective_sample;
    perspective_center = _out.perspective_center;
}


/////////////////////////////////////
// Entry point: "frag_main" (frag) //
/////////////////////////////////////
#version 460

struct FragmentInput
{
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

layout(location = 0) flat in uint _flat;
layout(location = 2) flat in uint flat_either;
layout(location = 3) noperspective in float _linear;
layout(location = 4) noperspective centroid in vec2 linear_centroid;
layout(location = 6) noperspective sample in vec3 linear_sample;
layout(location = 7) noperspective in vec3 linear_center;
layout(location = 8) in vec4 perspective;
layout(location = 9) centroid in float perspective_centroid;
layout(location = 10) sample in float perspective_sample;
layout(location = 11) in float perspective_center;

void main()
{
}

