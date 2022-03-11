static const float3 c_ambient = float3(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
static const uint c_max_lights = 10;

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
    linear float3 world_normal : LOC0;
    linear float4 world_position : LOC1;
};

struct Light {
    row_major float4x4 proj;
    float4 pos;
    float4 color;
};

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
    float _expr28 = t_shadow.SampleCmpLevelZero(sampler_shadow, float3(light_local, int(light_id)), (homogeneous_coords.z * proj_correction));
    return _expr28;
}

VertexOutput_vs_main vs_main(int4 position : LOC0, int4 normal : LOC1)
{
    VertexOutput out_ = (VertexOutput)0;

    float4x4 w = u_entity.world;
    float4x4 _expr7 = u_entity.world;
    float4 world_pos = mul(float4(position), _expr7);
    out_.world_normal = mul(float3(normal.xyz), float3x3(w[0].xyz, w[1].xyz, w[2].xyz));
    out_.world_position = world_pos;
    float4x4 _expr25 = u_globals.view_proj;
    out_.proj_position = mul(world_pos, _expr25);
    VertexOutput _expr27 = out_;
    const VertexOutput vertexoutput = _expr27;
    const VertexOutput_vs_main vertexoutput_1 = { vertexoutput.world_normal, vertexoutput.world_position, vertexoutput.proj_position };
    return vertexoutput_1;
}

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    VertexOutput in_ = { fragmentinput_fs_main.proj_position_1, fragmentinput_fs_main.world_normal_1, fragmentinput_fs_main.world_position_1 };
    float3 color = float3(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
    uint i = 0u;

    float3 normal_1 = normalize(in_.world_normal);
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
        uint _expr20 = i;
        i = (_expr20 + 1u);
        }
        loop_init = false;
        uint _expr14 = i;
        uint _expr17 = u_globals.num_lights.x;
        if ((_expr14 < min(_expr17, c_max_lights))) {
        } else {
            break;
        }
        uint _expr23 = i;
        Light light = {float4x4(asfloat(s_lights.Load4(_expr23*96+0+0)), asfloat(s_lights.Load4(_expr23*96+0+16)), asfloat(s_lights.Load4(_expr23*96+0+32)), asfloat(s_lights.Load4(_expr23*96+0+48))), asfloat(s_lights.Load4(_expr23*96+64)), asfloat(s_lights.Load4(_expr23*96+80))};
        uint _expr26 = i;
        const float _e30 = fetch_shadow(_expr26, mul(in_.world_position, light.proj));
        float3 light_dir = normalize((light.pos.xyz - in_.world_position.xyz));
        float diffuse = max(0.0, dot(normal_1, light_dir));
        float3 _expr40 = color;
        color = (_expr40 + ((_e30 * diffuse) * light.color.xyz));
    }
    float3 _expr46 = color;
    float4 _expr50 = u_entity.color;
    return (float4(_expr46, 1.0) * _expr50);
}

float4 fs_main_without_storage(FragmentInput_fs_main_without_storage fragmentinput_fs_main_without_storage) : SV_Target0
{
    VertexOutput in_1 = { fragmentinput_fs_main_without_storage.proj_position_2, fragmentinput_fs_main_without_storage.world_normal_2, fragmentinput_fs_main_without_storage.world_position_2 };
    float3 color_1 = float3(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
    uint i_1 = 0u;

    float3 normal_2 = normalize(in_1.world_normal);
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
        uint _expr20 = i_1;
        i_1 = (_expr20 + 1u);
        }
        loop_init_1 = false;
        uint _expr14 = i_1;
        uint _expr17 = u_globals.num_lights.x;
        if ((_expr14 < min(_expr17, c_max_lights))) {
        } else {
            break;
        }
        uint _expr23 = i_1;
        Light light_1 = u_lights[_expr23];
        uint _expr26 = i_1;
        const float _e30 = fetch_shadow(_expr26, mul(in_1.world_position, light_1.proj));
        float3 light_dir_1 = normalize((light_1.pos.xyz - in_1.world_position.xyz));
        float diffuse_1 = max(0.0, dot(normal_2, light_dir_1));
        float3 _expr40 = color_1;
        color_1 = (_expr40 + ((_e30 * diffuse_1) * light_1.color.xyz));
    }
    float3 _expr46 = color_1;
    float4 _expr50 = u_entity.color;
    return (float4(_expr46, 1.0) * _expr50);
}
