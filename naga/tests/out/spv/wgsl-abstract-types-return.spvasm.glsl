#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

int _10()
{
    return 1;
}

uint _15()
{
    return 1u;
}

float _20()
{
    return 1.0;
}

float _25()
{
    return 1.0;
}

vec2 _28()
{
    return vec2(1.0);
}

float[4] _33()
{
    return float[](1.0, 1.0, 1.0, 1.0);
}

float _38()
{
    return 1.0;
}

vec2 _41()
{
    return vec2(1.0);
}

void main()
{
}

