///////////////////////////////////////////////
// Entry point: "gen_terrain_compute" (comp) //
///////////////////////////////////////////////
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct ChunkData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
};

struct GenData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
    uint texture_size;
    uint start_index;
};

struct GenVertexOutput
{
    uint index;
    vec4 position;
    vec2 uv;
};

struct GenFragmentOutput
{
    uint vert_component;
    uint index;
};

struct Camera
{
    vec4 view_pos;
    mat4 view_proj;
};

struct Light
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 normal;
    vec3 world_pos;
};

layout(set = 0, binding = 0, std140) uniform chunk_data
{
    ChunkData _m0;
} chunk_data_1;

layout(set = 0, binding = 1, std430) buffer VertexBuffer
{
    Vertex data[];
} vertices;

layout(set = 0, binding = 2, std430) buffer IndexBuffer
{
    uint data[];
} indices;

layout(set = 0, binding = 0, std140) uniform gen_data
{
    GenData _m0;
} gen_data_1;

layout(set = 0, binding = 0, std140) uniform camera
{
    Camera _m0;
} camera_1;

layout(set = 1, binding = 0, std140) uniform light
{
    Light _m0;
} light_1;

layout(set = 2, binding = 0) uniform texture2D t_diffuse;
layout(set = 2, binding = 1) uniform sampler s_diffuse;
layout(set = 2, binding = 2) uniform texture2D t_normal;
layout(set = 2, binding = 3) uniform sampler s_normal;

uint naga_div(uint lhs, uint rhs)
{
    return lhs / ((rhs == 0u) ? 1u : rhs);
}

vec2 index_to_p(uint vert_index, uvec2 chunk_size, ivec2 chunk_corner)
{
    float _328 = float(vert_index);
    float _331 = float(chunk_size.x + 1u);
    return vec2(_328 - _331 * trunc(_328 / _331), float(naga_div(vert_index, chunk_size.x + 1u))) + vec2(chunk_corner);
}

vec3 permute3(vec3 x)
{
    vec3 _63 = ((x * 34.0) + vec3(1.0)) * x;
    return _63 - vec3(289.0) * trunc(_63 / vec3(289.0));
}

float snoise2(vec2 v)
{
    vec2 i1 = vec2(0.0);
    vec3 m = vec3(0.0);
    vec2 i = vec2(0.0);
    vec4 x12 = vec4(0.0);
    i = floor(v + vec2(dot(v, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).yy)));
    vec2 _103 = i;
    vec2 _105 = i;
    vec2 _109 = (v - _103) + vec2(dot(_105, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xx));
    i1 = mix(vec2(1.0, 0.0), vec2(0.0, 1.0), bvec2(_109.x < _109.y));
    x12 = (_109.xyxy + vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xxzz) - vec4(i1, 0.0, 0.0);
    i = i - vec2(289.0) * trunc(i / vec2(289.0));
    m = max(vec3(0.5) - vec3(dot(_109, _109), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.0));
    m *= m;
    m *= m;
    vec3 _169 = (fract(permute3((permute3(vec3(i.y) + vec3(0.0, i1.y, 1.0)) + vec3(i.x)) + vec3(0.0, i1.x, 1.0)) * vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).www) * 2.0) - vec3(1.0);
    vec3 _171 = abs(_169) - vec3(0.5);
    vec3 _174 = _169 - floor(_169 + vec3(0.5));
    m *= (vec3(1.792842864990234375) - (((_174 * _174) + (_171 * _171)) * 0.8537347316741943359375));
    return 130.0 * dot(m, vec3((_174.x * _109.x) + (_171.x * _109.y), (_174.yz * x12.xz) + (_171.yz * x12.yw)));
}

float fbm(vec2 p)
{
    float v = 0.0;
    uint i = 0u;
    vec2 x = vec2(0.0);
    float a = 0.5;
    uvec2 loop_bound = uvec2(4294967295u);
    x = p * 0.00999999977648258209228515625;
    mat2 _227 = mat2(vec2(0.877582550048828125, 0.47942554950714111328125), vec2(-0.47942554950714111328125, 0.877582550048828125));
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (!(i < 5u))
        {
            break;
        }
        v += (a * snoise2(x));
        x = ((_227 * x) * 2.0) + vec2(100.0);
        a *= 0.5;
        i++;
        continue;
    }
    return v;
}

vec3 terrain_point(vec2 p, vec2 min_max_height)
{
    return vec3(p.x, mix(min_max_height.x, min_max_height.y, fbm(p)), p.y);
}

Vertex terrain_vertex(vec2 p, vec2 min_max_height)
{
    vec3 _293 = terrain_point(p, min_max_height);
    return Vertex(_293, (normalize(cross(terrain_point(p + vec2(0.0, 0.100000001490116119384765625), min_max_height) - _293, terrain_point(p + vec2(0.100000001490116119384765625, 0.0), min_max_height) - _293)) + normalize(cross(terrain_point(p + vec2(0.0, -0.100000001490116119384765625), min_max_height) - _293, terrain_point(p + vec2(-0.100000001490116119384765625, 0.0), min_max_height) - _293))) * 0.5);
}

void main()
{
    vertices.data[gl_GlobalInvocationID.x] = terrain_vertex(index_to_p(gl_GlobalInvocationID.x, chunk_data_1._m0.chunk_size, chunk_data_1._m0.chunk_corner), chunk_data_1._m0.min_max_height);
    uint _392 = gl_GlobalInvocationID.x * 6u;
    if (_392 >= ((chunk_data_1._m0.chunk_size.x * chunk_data_1._m0.chunk_size.y) * 6u))
    {
        return;
    }
    uint _407 = gl_GlobalInvocationID.x + naga_div(gl_GlobalInvocationID.x, chunk_data_1._m0.chunk_size.x);
    uint _412 = (_407 + chunk_data_1._m0.chunk_size.x) + 1u;
    uint _413 = _412 + 1u;
    indices.data[_392] = _407;
    indices.data[_392 + 1u] = _412;
    indices.data[_392 + 2u] = _413;
    indices.data[_392 + 3u] = _407;
    indices.data[_392 + 4u] = _413;
    indices.data[_392 + 5u] = _407 + 1u;
}


//////////////////////////////////////////////
// Entry point: "gen_terrain_vertex" (vert) //
//////////////////////////////////////////////
#version 460

struct ChunkData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
};

struct GenData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
    uint texture_size;
    uint start_index;
};

struct GenVertexOutput
{
    uint index;
    vec4 position;
    vec2 uv;
};

struct GenFragmentOutput
{
    uint vert_component;
    uint index;
};

struct Camera
{
    vec4 view_pos;
    mat4 view_proj;
};

struct Light
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 normal;
    vec3 world_pos;
};

layout(set = 0, binding = 0, std140) uniform chunk_data
{
    ChunkData _m0;
} chunk_data_1;

layout(set = 0, binding = 1, std430) buffer VertexBuffer
{
    Vertex data[];
} vertices;

layout(set = 0, binding = 2, std430) buffer IndexBuffer
{
    uint data[];
} indices;

layout(set = 0, binding = 0, std140) uniform gen_data
{
    GenData _m0;
} gen_data_1;

layout(set = 0, binding = 0, std140) uniform camera
{
    Camera _m0;
} camera_1;

layout(set = 1, binding = 0, std140) uniform light
{
    Light _m0;
} light_1;

layout(set = 2, binding = 0) uniform texture2D t_diffuse;
layout(set = 2, binding = 1) uniform sampler s_diffuse;
layout(set = 2, binding = 2) uniform texture2D t_normal;
layout(set = 2, binding = 3) uniform sampler s_normal;

layout(location = 0) flat out uint index;
layout(location = 1) out vec2 uv;

uint naga_div(uint lhs, uint rhs)
{
    return lhs / ((rhs == 0u) ? 1u : rhs);
}

uint naga_mod(uint lhs, uint rhs)
{
    return lhs % ((rhs == 0u) ? 1u : rhs);
}

void main()
{
    vec2 _458 = vec2(float(naga_mod(naga_div(uint(gl_VertexIndex) + 2u, 3u), 2u)), float(naga_mod(naga_div(uint(gl_VertexIndex) + 1u, 3u), 2u)));
    GenVertexOutput _479 = GenVertexOutput(uint(clamp((_458.x * float(gen_data_1._m0.texture_size)) + (_458.y * float(gen_data_1._m0.texture_size)), 0.0, 4294967040.0)) + gen_data_1._m0.start_index, vec4(vec2(-1.0) + (_458 * 2.0), 0.0, 1.0), _458);
    index = _479.index;
    gl_Position = _479.position;
    uv = _479.uv;
}


////////////////////////////////////////////////
// Entry point: "gen_terrain_fragment" (frag) //
////////////////////////////////////////////////
#version 460

struct ChunkData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
};

struct GenData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
    uint texture_size;
    uint start_index;
};

struct GenVertexOutput
{
    uint index;
    vec4 position;
    vec2 uv;
};

struct GenFragmentOutput
{
    uint vert_component;
    uint index;
};

struct Camera
{
    vec4 view_pos;
    mat4 view_proj;
};

struct Light
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 normal;
    vec3 world_pos;
};

layout(set = 0, binding = 0, std140) uniform chunk_data
{
    ChunkData _m0;
} chunk_data_1;

layout(set = 0, binding = 1, std430) buffer VertexBuffer
{
    Vertex data[];
} vertices;

layout(set = 0, binding = 2, std430) buffer IndexBuffer
{
    uint data[];
} indices;

layout(set = 0, binding = 0, std140) uniform gen_data
{
    GenData _m0;
} gen_data_1;

layout(set = 0, binding = 0, std140) uniform camera
{
    Camera _m0;
} camera_1;

layout(set = 1, binding = 0, std140) uniform light
{
    Light _m0;
} light_1;

layout(set = 2, binding = 0) uniform texture2D t_diffuse;
layout(set = 2, binding = 1) uniform sampler s_diffuse;
layout(set = 2, binding = 2) uniform texture2D t_normal;
layout(set = 2, binding = 3) uniform sampler s_normal;

layout(location = 0) flat in uint index;
layout(location = 1) in vec2 uv;
layout(location = 0) out uint vert_component;
layout(location = 1) out uint index_1;

uint naga_mod(uint lhs, uint rhs)
{
    return lhs % ((rhs == 0u) ? 1u : rhs);
}

uint naga_div(uint lhs, uint rhs)
{
    return lhs / ((rhs == 0u) ? 1u : rhs);
}

vec2 index_to_p(uint vert_index, uvec2 chunk_size, ivec2 chunk_corner)
{
    float _328 = float(vert_index);
    float _331 = float(chunk_size.x + 1u);
    return vec2(_328 - _331 * trunc(_328 / _331), float(naga_div(vert_index, chunk_size.x + 1u))) + vec2(chunk_corner);
}

vec3 permute3(vec3 x)
{
    vec3 _63 = ((x * 34.0) + vec3(1.0)) * x;
    return _63 - vec3(289.0) * trunc(_63 / vec3(289.0));
}

float snoise2(vec2 v)
{
    vec2 i1 = vec2(0.0);
    vec3 m = vec3(0.0);
    vec2 i = vec2(0.0);
    vec4 x12 = vec4(0.0);
    i = floor(v + vec2(dot(v, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).yy)));
    vec2 _103 = i;
    vec2 _105 = i;
    vec2 _109 = (v - _103) + vec2(dot(_105, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xx));
    i1 = mix(vec2(1.0, 0.0), vec2(0.0, 1.0), bvec2(_109.x < _109.y));
    x12 = (_109.xyxy + vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xxzz) - vec4(i1, 0.0, 0.0);
    i = i - vec2(289.0) * trunc(i / vec2(289.0));
    m = max(vec3(0.5) - vec3(dot(_109, _109), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.0));
    m *= m;
    m *= m;
    vec3 _169 = (fract(permute3((permute3(vec3(i.y) + vec3(0.0, i1.y, 1.0)) + vec3(i.x)) + vec3(0.0, i1.x, 1.0)) * vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).www) * 2.0) - vec3(1.0);
    vec3 _171 = abs(_169) - vec3(0.5);
    vec3 _174 = _169 - floor(_169 + vec3(0.5));
    m *= (vec3(1.792842864990234375) - (((_174 * _174) + (_171 * _171)) * 0.8537347316741943359375));
    return 130.0 * dot(m, vec3((_174.x * _109.x) + (_171.x * _109.y), (_174.yz * x12.xz) + (_171.yz * x12.yw)));
}

float fbm(vec2 p)
{
    float v = 0.0;
    uint i = 0u;
    vec2 x = vec2(0.0);
    float a = 0.5;
    uvec2 loop_bound = uvec2(4294967295u);
    x = p * 0.00999999977648258209228515625;
    mat2 _227 = mat2(vec2(0.877582550048828125, 0.47942554950714111328125), vec2(-0.47942554950714111328125, 0.877582550048828125));
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (!(i < 5u))
        {
            break;
        }
        v += (a * snoise2(x));
        x = ((_227 * x) * 2.0) + vec2(100.0);
        a *= 0.5;
        i++;
        continue;
    }
    return v;
}

vec3 terrain_point(vec2 p, vec2 min_max_height)
{
    return vec3(p.x, mix(min_max_height.x, min_max_height.y, fbm(p)), p.y);
}

Vertex terrain_vertex(vec2 p, vec2 min_max_height)
{
    vec3 _293 = terrain_point(p, min_max_height);
    return Vertex(_293, (normalize(cross(terrain_point(p + vec2(0.0, 0.100000001490116119384765625), min_max_height) - _293, terrain_point(p + vec2(0.100000001490116119384765625, 0.0), min_max_height) - _293)) + normalize(cross(terrain_point(p + vec2(0.0, -0.100000001490116119384765625), min_max_height) - _293, terrain_point(p + vec2(-0.100000001490116119384765625, 0.0), min_max_height) - _293))) * 0.5);
}

void main()
{
    float vert_component_1 = 0.0;
    uint index_2 = 0u;
    GenVertexOutput _484 = GenVertexOutput(index, gl_FragCoord, uv);
    uint _521 = uint(clamp((_484.uv.x * float(gen_data_1._m0.texture_size)) + (_484.uv.y * float(gen_data_1._m0.texture_size * gen_data_1._m0.texture_size)), 0.0, 4294967040.0)) + gen_data_1._m0.start_index;
    uint _526 = uint(clamp(floor(float(_521) / 6.0), 0.0, 4294967040.0));
    uint _527 = naga_mod(_521, 6u);
    Vertex _535 = terrain_vertex(index_to_p(_526, gen_data_1._m0.chunk_size, gen_data_1._m0.chunk_corner), gen_data_1._m0.min_max_height);
    switch (_527)
    {
        case 0u:
        {
            vert_component_1 = _535.position.x;
            break;
        }
        case 1u:
        {
            vert_component_1 = _535.position.y;
            break;
        }
        case 2u:
        {
            vert_component_1 = _535.position.z;
            break;
        }
        case 3u:
        {
            vert_component_1 = _535.normal.x;
            break;
        }
        case 4u:
        {
            vert_component_1 = _535.normal.y;
            break;
        }
        case 5u:
        {
            vert_component_1 = _535.normal.z;
            break;
        }
        default:
        {
            break;
        }
    }
    uint _559 = _526 + naga_div(_526, gen_data_1._m0.chunk_size.x);
    uint _564 = (_559 + gen_data_1._m0.chunk_size.x) + 1u;
    switch (_527)
    {
        case 0u:
        case 3u:
        {
            index_2 = _559;
            break;
        }
        case 2u:
        case 4u:
        {
            index_2 = _564 + 1u;
            break;
        }
        case 1u:
        {
            index_2 = _564;
            break;
        }
        case 5u:
        {
            index_2 = _559 + 1u;
            break;
        }
        default:
        {
            break;
        }
    }
    index_2 = _484.index;
    GenFragmentOutput _576 = GenFragmentOutput(floatBitsToUint(vert_component_1), index_2);
    vert_component = _576.vert_component;
    index_1 = _576.index;
}


///////////////////////////////////
// Entry point: "vs_main" (vert) //
///////////////////////////////////
#version 460

struct ChunkData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
};

struct GenData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
    uint texture_size;
    uint start_index;
};

struct GenVertexOutput
{
    uint index;
    vec4 position;
    vec2 uv;
};

struct GenFragmentOutput
{
    uint vert_component;
    uint index;
};

struct Camera
{
    vec4 view_pos;
    mat4 view_proj;
};

struct Light
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 normal;
    vec3 world_pos;
};

layout(set = 0, binding = 0, std140) uniform chunk_data
{
    ChunkData _m0;
} chunk_data_1;

layout(set = 0, binding = 1, std430) buffer VertexBuffer
{
    Vertex data[];
} vertices;

layout(set = 0, binding = 2, std430) buffer IndexBuffer
{
    uint data[];
} indices;

layout(set = 0, binding = 0, std140) uniform gen_data
{
    GenData _m0;
} gen_data_1;

layout(set = 0, binding = 0, std140) uniform camera
{
    Camera _m0;
} camera_1;

layout(set = 1, binding = 0, std140) uniform light
{
    Light _m0;
} light_1;

layout(set = 2, binding = 0) uniform texture2D t_diffuse;
layout(set = 2, binding = 1) uniform sampler s_diffuse;
layout(set = 2, binding = 2) uniform texture2D t_normal;
layout(set = 2, binding = 3) uniform sampler s_normal;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 normal_1;
layout(location = 1) out vec3 world_pos;

void main()
{
    Vertex _580 = Vertex(position, normal);
    VertexOutput _602 = VertexOutput(camera_1._m0.view_proj * vec4(_580.position, 1.0), _580.normal, _580.position);
    gl_Position = _602.clip_position;
    normal_1 = _602.normal;
    world_pos = _602.world_pos;
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460

struct ChunkData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
};

struct Vertex
{
    vec3 position;
    vec3 normal;
};

struct GenData
{
    uvec2 chunk_size;
    ivec2 chunk_corner;
    vec2 min_max_height;
    uint texture_size;
    uint start_index;
};

struct GenVertexOutput
{
    uint index;
    vec4 position;
    vec2 uv;
};

struct GenFragmentOutput
{
    uint vert_component;
    uint index;
};

struct Camera
{
    vec4 view_pos;
    mat4 view_proj;
};

struct Light
{
    vec3 position;
    vec3 color;
};

struct VertexOutput
{
    vec4 clip_position;
    vec3 normal;
    vec3 world_pos;
};

layout(set = 0, binding = 0, std140) uniform chunk_data
{
    ChunkData _m0;
} chunk_data_1;

layout(set = 0, binding = 1, std430) buffer VertexBuffer
{
    Vertex data[];
} vertices;

layout(set = 0, binding = 2, std430) buffer IndexBuffer
{
    uint data[];
} indices;

layout(set = 0, binding = 0, std140) uniform gen_data
{
    GenData _m0;
} gen_data_1;

layout(set = 0, binding = 0, std140) uniform camera
{
    Camera _m0;
} camera_1;

layout(set = 1, binding = 0, std140) uniform light
{
    Light _m0;
} light_1;

layout(set = 2, binding = 0) uniform texture2D t_diffuse;
layout(set = 2, binding = 1) uniform sampler s_diffuse;
layout(set = 2, binding = 2) uniform texture2D t_normal;
layout(set = 2, binding = 3) uniform sampler s_normal;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_pos;
layout(location = 0) out vec4 _614;

vec3 permute3(vec3 x)
{
    vec3 _63 = ((x * 34.0) + vec3(1.0)) * x;
    return _63 - vec3(289.0) * trunc(_63 / vec3(289.0));
}

float snoise2(vec2 v)
{
    vec2 i1 = vec2(0.0);
    vec3 m = vec3(0.0);
    vec2 i = vec2(0.0);
    vec4 x12 = vec4(0.0);
    i = floor(v + vec2(dot(v, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).yy)));
    vec2 _103 = i;
    vec2 _105 = i;
    vec2 _109 = (v - _103) + vec2(dot(_105, vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xx));
    i1 = mix(vec2(1.0, 0.0), vec2(0.0, 1.0), bvec2(_109.x < _109.y));
    x12 = (_109.xyxy + vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).xxzz) - vec4(i1, 0.0, 0.0);
    i = i - vec2(289.0) * trunc(i / vec2(289.0));
    m = max(vec3(0.5) - vec3(dot(_109, _109), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.0));
    m *= m;
    m *= m;
    vec3 _169 = (fract(permute3((permute3(vec3(i.y) + vec3(0.0, i1.y, 1.0)) + vec3(i.x)) + vec3(0.0, i1.x, 1.0)) * vec4(0.211324870586395263671875, 0.3660254180431365966796875, -0.57735025882720947265625, 0.024390242993831634521484375).www) * 2.0) - vec3(1.0);
    vec3 _171 = abs(_169) - vec3(0.5);
    vec3 _174 = _169 - floor(_169 + vec3(0.5));
    m *= (vec3(1.792842864990234375) - (((_174 * _174) + (_171 * _171)) * 0.8537347316741943359375));
    return 130.0 * dot(m, vec3((_174.x * _109.x) + (_171.x * _109.y), (_174.yz * x12.xz) + (_171.yz * x12.yw)));
}

vec3 color23(vec2 p)
{
    return vec3((snoise2(p) * 0.5) + 0.5, (snoise2(p + vec2(23.0, 32.0)) * 0.5) + 0.5, (snoise2(p + vec2(-43.0, 3.0)) * 0.5) + 0.5);
}

void main()
{
    vec3 color = vec3(0.0);
    VertexOutput _607 = VertexOutput(gl_FragCoord, normal, world_pos);
    color = smoothstep(vec3(0.0), vec3(0.100000001490116119384765625), fract(_607.world_pos));
    color = mix(vec3(0.5, 0.100000001490116119384765625, 0.699999988079071044921875), vec3(0.20000000298023223876953125), vec3((color.x * color.y) * color.z));
    vec3 _654 = normalize(light_1._m0.position - _607.world_pos);
    _614 = vec4((((light_1._m0.color * 0.100000001490116119384765625) + (light_1._m0.color * max(dot(_607.normal, _654), 0.0))) + (light_1._m0.color * pow(max(dot(_607.normal, normalize(normalize(camera_1._m0.view_pos.xyz - _607.world_pos) + _654)), 0.0), 32.0))) * color, 1.0);
}

