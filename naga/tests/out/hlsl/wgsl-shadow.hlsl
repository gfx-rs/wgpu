struct Globals {
    row_major float4x4 view_proj;
    uint4 num_lights;
};

struct Entity {
    row_major float4x4 world;
    float4 color;
};

struct VertexOutput {
    float4 proj_position : SV_Position;
    float3 world_normal : LOC0;
    float4 world_position : LOC1;
};

struct Light {
    row_major float4x4 proj;
    float4 pos;
    float4 color;
};

static const float3 c_ambient = float3(0.05, 0.05, 0.05);
static const uint c_max_lights = 10u;

cbuffer u_globals : register(b0) { Globals u_globals; }
cbuffer u_entity : register(b0, space1) { Entity u_entity; }
ByteAddressBuffer s_lights : register(t1);
cbuffer u_lights : register(b1) { Light u_lights[10]; }
Texture2DArray<float> t_shadow : register(t2);
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const SamplerComparisonState sampler_shadow = nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[3]];

struct VertexOutput_vs_main {
    float3 world_normal : LOC0;
    float4 world_position : LOC1;
    float4 proj_position : SV_Position;
};

struct FragmentInput_fs_main {
    float3 world_normal_1 : LOC0;
    float4 world_position_1 : LOC1;
    float4 proj_position_1 : SV_Position;
};

struct FragmentInput_fs_main_without_storage {
    float3 world_normal_2 : LOC0;
    float4 world_position_2 : LOC1;
    float4 proj_position_2 : SV_Position;
};

VertexOutput_vs_main vs_main(int4 position : LOC0, int4 normal : LOC1)
{
    VertexOutput out_ = (VertexOutput)0;

    float4x4 w = u_entity.world;
    float4x4 _e7 = u_entity.world;
    float4 world_pos = mul(float4(position), _e7);
    out_.world_normal = mul(float3(normal.xyz), float3x3(w[0].xyz, w[1].xyz, w[2].xyz));
    out_.world_position = world_pos;
    float4x4 _e26 = u_globals.view_proj;
    out_.proj_position = mul(world_pos, _e26);
    VertexOutput _e28 = out_;
    const VertexOutput vertexoutput = _e28;
    const VertexOutput_vs_main vertexoutput_1 = { vertexoutput.world_normal, vertexoutput.world_position, vertexoutput.proj_position };
    return vertexoutput_1;
}

Light ConstructLight(float4x4 arg0, float4 arg1, float4 arg2) {
    Light ret = (Light)0;
    ret.proj = arg0;
    ret.pos = arg1;
    ret.color = arg2;
    return ret;
}

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    VertexOutput in_ = { float4(fragmentinput_fs_main.proj_position_1.xyz, 1.0 / fragmentinput_fs_main.proj_position_1.w), fragmentinput_fs_main.world_normal_1, fragmentinput_fs_main.world_position_1 };
    float3 color = c_ambient;
    uint i = 0u;
    uint light_id = (uint)0;
    float4 homogeneous_coords = (float4)0;
    float local = (float)0;
    uint2 loop_bound = (4294967295u).xx;

    float3 normal_1 = normalize(in_.world_normal);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e77 = i;
            i = (_e77 + 1u);
        }
        loop_init = false;
        {
            uint2 _e6 = loop_bound;
            loop_bound = (_e6 - uint2(((_e6.y == 0u) ? 1u : 0u), 1u));
            if (all((_e6 == (0u).xx))) {
                break;
            }
        }
        uint _e16 = i;
        uint _e20 = u_globals.num_lights.x;
        if ((_e16 < min(_e20, c_max_lights))) {
        } else {
            break;
        }
        {
            uint _e25 = i;
            Light light = ConstructLight(float4x4(asfloat(s_lights.Load4(_e25*96+0+0)), asfloat(s_lights.Load4(_e25*96+0+16)), asfloat(s_lights.Load4(_e25*96+0+32)), asfloat(s_lights.Load4(_e25*96+0+48))), asfloat(s_lights.Load4(_e25*96+64)), asfloat(s_lights.Load4(_e25*96+80)));
            uint _e28 = i;
            light_id = _e28;
            uint _e33 = light_id;
            homogeneous_coords = mul(in_.world_position, light.proj);
            float4 _e35 = homogeneous_coords;
            bool should_continue = false;
            do {
                if ((_e35.w <= 0.0)) {
                    local = 1.0;
                    break;
                }
                float _e44 = (1.0 / _e35.w);
                float _e57 = t_shadow.SampleCmpLevelZero(sampler_shadow, float3((((_e35.xy * float2(0.5, -0.5)) * _e44) + float2(0.5, 0.5)), int(_e33)), (_e35.z * _e44));
                local = _e57;
            } while(false);
            float shadow = local;
            float3 light_dir = normalize((light.pos.xyz - in_.world_position.xyz));
            float diffuse = max(0.0, dot(normal_1, light_dir));
            float3 _e70 = color;
            color = (_e70 + ((shadow * diffuse) * light.color.xyz));
        }
    }
    float3 _e79 = color;
    float4 _e84 = u_entity.color;
    return (float4(_e79, 1.0) * _e84);
}

float4 fs_main_without_storage(FragmentInput_fs_main_without_storage fragmentinput_fs_main_without_storage) : SV_Target0
{
    VertexOutput in_1 = { float4(fragmentinput_fs_main_without_storage.proj_position_2.xyz, 1.0 / fragmentinput_fs_main_without_storage.proj_position_2.w), fragmentinput_fs_main_without_storage.world_normal_2, fragmentinput_fs_main_without_storage.world_position_2 };
    float3 color_1 = c_ambient;
    uint i_1 = 0u;
    uint light_id_1 = (uint)0;
    float4 homogeneous_coords_1 = (float4)0;
    float local_1 = (float)0;
    uint2 loop_bound_1 = (4294967295u).xx;

    float3 normal_2 = normalize(in_1.world_normal);
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e77 = i_1;
            i_1 = (_e77 + 1u);
        }
        loop_init_1 = false;
        {
            uint2 _e6 = loop_bound_1;
            loop_bound_1 = (_e6 - uint2(((_e6.y == 0u) ? 1u : 0u), 1u));
            if (all((_e6 == (0u).xx))) {
                break;
            }
        }
        uint _e16 = i_1;
        uint _e20 = u_globals.num_lights.x;
        if ((_e16 < min(_e20, c_max_lights))) {
        } else {
            break;
        }
        {
            uint _e25 = i_1;
            Light light_1 = u_lights[_e25];
            uint _e28 = i_1;
            light_id_1 = _e28;
            uint _e33 = light_id_1;
            homogeneous_coords_1 = mul(in_1.world_position, light_1.proj);
            float4 _e35 = homogeneous_coords_1;
            bool should_continue_1 = false;
            do {
                if ((_e35.w <= 0.0)) {
                    local_1 = 1.0;
                    break;
                }
                float _e44 = (1.0 / _e35.w);
                float _e57 = t_shadow.SampleCmpLevelZero(sampler_shadow, float3((((_e35.xy * float2(0.5, -0.5)) * _e44) + float2(0.5, 0.5)), int(_e33)), (_e35.z * _e44));
                local_1 = _e57;
            } while(false);
            float shadow_1 = local_1;
            float3 light_dir_1 = normalize((light_1.pos.xyz - in_1.world_position.xyz));
            float diffuse_1 = max(0.0, dot(normal_2, light_dir_1));
            float3 _e70 = color_1;
            color_1 = (_e70 + ((shadow_1 * diffuse_1) * light_1.color.xyz));
        }
    }
    float3 _e79 = color_1;
    float4 _e84 = u_entity.color;
    return (float4(_e79, 1.0) * _e84);
}
