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
SamplerComparisonState sampler_shadow : register(s3);

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

float fetch_shadow(uint light_id, float4 homogeneous_coords)
{
    if ((homogeneous_coords.w <= 0.0)) {
        return 1.0;
    }
    float2 flip_correction = float2(0.5, -0.5);
    float proj_correction = (1.0 / homogeneous_coords.w);
    float2 light_local = (((homogeneous_coords.xy * flip_correction) * proj_correction) + float2(0.5, 0.5));
    float _expr24 = t_shadow.SampleCmpLevelZero(sampler_shadow, float3(light_local, int(light_id)), (homogeneous_coords.z * proj_correction));
    return _expr24;
}

VertexOutput_vs_main vs_main(int4 position : LOC0, int4 normal : LOC1)
{
    VertexOutput out_ = (VertexOutput)0;

    float4x4 w = u_entity.world;
    float4x4 _expr7 = u_entity.world;
    float4 world_pos = mul(float4(position), _expr7);
    out_.world_normal = mul(float3(normal.xyz), float3x3(w[0].xyz, w[1].xyz, w[2].xyz));
    out_.world_position = world_pos;
    float4x4 _expr26 = u_globals.view_proj;
    out_.proj_position = mul(world_pos, _expr26);
    VertexOutput _expr28 = out_;
    const VertexOutput vertexoutput = _expr28;
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
    VertexOutput in_ = { fragmentinput_fs_main.proj_position_1, fragmentinput_fs_main.world_normal_1, fragmentinput_fs_main.world_position_1 };
    float3 color = (float3)0;
    uint i = (uint)0;

    float3 normal_1 = normalize(in_.world_normal);
    color = c_ambient;
    i = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _expr40 = i;
            i = (_expr40 + 1u);
        }
        loop_init = false;
        uint _expr7 = i;
        uint _expr11 = u_globals.num_lights.x;
        if ((_expr7 < min(_expr11, c_max_lights))) {
        } else {
            break;
        }
        {
            uint _expr16 = i;
            Light light = ConstructLight(float4x4(asfloat(s_lights.Load4(_expr16*96+0+0)), asfloat(s_lights.Load4(_expr16*96+0+16)), asfloat(s_lights.Load4(_expr16*96+0+32)), asfloat(s_lights.Load4(_expr16*96+0+48))), asfloat(s_lights.Load4(_expr16*96+64)), asfloat(s_lights.Load4(_expr16*96+80)));
            uint _expr19 = i;
            const float _e23 = fetch_shadow(_expr19, mul(in_.world_position, light.proj));
            float3 light_dir = normalize((light.pos.xyz - in_.world_position.xyz));
            float diffuse = max(0.0, dot(normal_1, light_dir));
            float3 _expr37 = color;
            color = (_expr37 + ((_e23 * diffuse) * light.color.xyz));
        }
    }
    float3 _expr42 = color;
    float4 _expr47 = u_entity.color;
    return (float4(_expr42, 1.0) * _expr47);
}

float4 fs_main_without_storage(FragmentInput_fs_main_without_storage fragmentinput_fs_main_without_storage) : SV_Target0
{
    VertexOutput in_1 = { fragmentinput_fs_main_without_storage.proj_position_2, fragmentinput_fs_main_without_storage.world_normal_2, fragmentinput_fs_main_without_storage.world_position_2 };
    float3 color_1 = (float3)0;
    uint i_1 = (uint)0;

    float3 normal_2 = normalize(in_1.world_normal);
    color_1 = c_ambient;
    i_1 = 0u;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _expr40 = i_1;
            i_1 = (_expr40 + 1u);
        }
        loop_init_1 = false;
        uint _expr7 = i_1;
        uint _expr11 = u_globals.num_lights.x;
        if ((_expr7 < min(_expr11, c_max_lights))) {
        } else {
            break;
        }
        {
            uint _expr16 = i_1;
            Light light_1 = u_lights[_expr16];
            uint _expr19 = i_1;
            const float _e23 = fetch_shadow(_expr19, mul(in_1.world_position, light_1.proj));
            float3 light_dir_1 = normalize((light_1.pos.xyz - in_1.world_position.xyz));
            float diffuse_1 = max(0.0, dot(normal_2, light_dir_1));
            float3 _expr37 = color_1;
            color_1 = (_expr37 + ((_e23 * diffuse_1) * light_1.color.xyz));
        }
    }
    float3 _expr42 = color_1;
    float4 _expr47 = u_entity.color;
    return (float4(_expr42, 1.0) * _expr47);
}
