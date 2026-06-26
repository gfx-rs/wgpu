#version 460

struct S
{
    vec3 a;
};

struct Test
{
    S a;
    float b;
};

struct Test2
{
    vec3 a[2];
    float b;
};

struct Test3
{
    mat4x3 a;
    float b;
};

layout(set = 0, binding = 0, std140) uniform input1
{
    Test _m0;
} input1_1;

layout(set = 0, binding = 1, std140) uniform input2
{
    Test2 _m0;
} input2_1;

layout(set = 0, binding = 2, std140) uniform input3
{
    Test3 _m0;
} input3_1;

void main()
{
    gl_Position = ((vec4(1.0) * input1_1._m0.b) * input2_1._m0.b) * input3_1._m0.b;
}

