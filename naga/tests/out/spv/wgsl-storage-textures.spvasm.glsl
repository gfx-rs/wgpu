//////////////////////////////////
// Entry point: "csLoad" (comp) //
//////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D _7;
layout(set = 0, binding = 1, rg32f) uniform readonly image2D _9;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D _11;
layout(set = 1, binding = 0, r32f) uniform writeonly image2D _13;
layout(set = 1, binding = 1, rg32f) uniform writeonly image2D _14;
layout(set = 1, binding = 2, rgba32f) uniform writeonly image2D _15;

void main()
{
}


///////////////////////////////////
// Entry point: "csStore" (comp) //
///////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r32f) uniform readonly image2D _7;
layout(set = 0, binding = 1, rg32f) uniform readonly image2D _9;
layout(set = 0, binding = 2, rgba32f) uniform readonly image2D _11;
layout(set = 1, binding = 0, r32f) uniform writeonly image2D _13;
layout(set = 1, binding = 1, rg32f) uniform writeonly image2D _14;
layout(set = 1, binding = 2, rgba32f) uniform writeonly image2D _15;

void main()
{
    imageStore(_13, ivec2(uvec2(0u)), vec4(0.0));
    imageStore(_14, ivec2(uvec2(0u)), vec4(0.0));
    imageStore(_15, ivec2(uvec2(0u)), vec4(0.0));
}

