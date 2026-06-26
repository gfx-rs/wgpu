#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    ivec2 _17 = ivec2(1, 2);
    vec2 _19 = vec2(0.0);
    _19 = mix(vec2(1.0, 0.0), vec2(0.0, 1.0), bvec2(_17.x < _17.y));
}

