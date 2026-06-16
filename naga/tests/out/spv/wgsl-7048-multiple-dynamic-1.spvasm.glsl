#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const vec3 _13[2] = vec3[](vec3(0.0), vec3(0.0));

void main()
{
    vec4 _17 = vec4(0.0);
    int _19 = 0;
    int _21 = 0;
    _17.x += (_13[_21].y * _13[_19].z);
}

