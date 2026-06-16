/////////////////////////////////////
// Entry point: "vert_main" (vert) //
/////////////////////////////////////
#version 460

struct VertexOutput
{
    vec2 uv;
    vec4 position;
};

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 uv_1;

void main()
{
    VertexOutput _32 = VertexOutput(uv, vec4(pos * 1.2000000476837158203125, 0.0, 1.0));
    uv_1 = _32.uv;
    gl_Position = _32.position;
    gl_Position.y = -gl_Position.y;
}


/////////////////////////////////////
// Entry point: "frag_main" (frag) //
/////////////////////////////////////
#version 460

struct VertexOutput
{
    vec2 uv;
    vec4 position;
};

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 _44;

void main()
{
    vec4 _51 = texture(sampler2D(u_texture, u_sampler), uv);
    if (_51.w == 0.0)
    {
        discard;
    }
    _44 = _51 * _51.w;
}


////////////////////////////////////
// Entry point: "fs_extra" (frag) //
////////////////////////////////////
#version 460

struct VertexOutput
{
    vec2 uv;
    vec4 position;
};

layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(location = 0) out vec4 _60;

void main()
{
    _60 = vec4(0.0, 0.5, 0.0, 0.5);
}

