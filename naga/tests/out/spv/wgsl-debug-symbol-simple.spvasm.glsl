///////////////////////////////////
// Entry point: "vs_main" (vert) //
///////////////////////////////////
#version 460

struct VertexInput
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 color;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 0) out vec3 color_1;

void main()
{
    VertexOutput _out = VertexOutput(vec4(0.0), vec3(0.0));
    VertexInput _13 = VertexInput(position, color);
    _out.color = _13.color;
    _out.clip_position = vec4(_13.position, 1.0);
    gl_Position = _out.clip_position;
    color_1 = _out.color;
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460

struct VertexInput
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 color;
};

layout(location = 0) in vec3 color;
layout(location = 0) out vec4 _49;

void main()
{
    int i = 0;
    uvec2 loop_bound = uvec2(0u);
    vec3 color_1 = vec3(0.0);
    float ii = 0.0;
    color_1 = VertexOutput(gl_FragCoord, color).color;
    loop_bound = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(loop_bound, uvec2(0u))))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (!(i < 10))
        {
            break;
        }
        ii = float(i);
        color_1.x += (ii * 0.001000000047497451305389404296875);
        color_1.y += (ii * 0.00200000009499490261077880859375);
        i++;
        continue;
    }
    _49 = vec4(color_1, 1.0);
}

