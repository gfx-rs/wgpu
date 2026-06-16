////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

ivec2 naga_mod(ivec2 lhs, ivec2 rhs)
{
    bvec2 _79 = equal(rhs, ivec2(0));
    bvec2 _84 = equal(lhs, ivec2(int(0x80000000)));
    bvec2 _85 = equal(rhs, ivec2(-1));
    bvec2 _86 = bvec2(_84.x && _85.x, _84.y && _85.y);
    return lhs - mix(rhs, ivec2(1), bvec2(_79.x || _86.x, _79.y || _86.y)) * (lhs / mix(rhs, ivec2(1), bvec2(_79.x || _86.x, _79.y || _86.y)));
}

void main()
{
    ivec2 _114 = naga_mod(ivec2(uvec2(imageSize(image_storage_src)) * gl_LocalInvocationID.xy), ivec2(10, 20));
    uvec4 _162 = imageLoad(image_storage_src, ivec2(uvec2(_114)));
    imageStore(image_dst, _114.x, (((texelFetch(image_mipmapped_src, _114, int(gl_LocalInvocationID.z)) + texelFetch(image_multisampled_src, _114, int(gl_LocalInvocationID.z))) + imageLoad(image_storage_src, _114)) + texelFetch(image_array_src, ivec3(_114, int(gl_LocalInvocationID.z)), int(gl_LocalInvocationID.z) + 1)) + texelFetch(image_array_src, ivec3(_114, int(gl_LocalInvocationID.z)), int(gl_LocalInvocationID.z) + 1));
    imageStore(image_dst, int(uint(_114.x)), (((texelFetch(image_mipmapped_src, ivec2(uvec2(_114)), int(gl_LocalInvocationID.z)) + texelFetch(image_multisampled_src, ivec2(uvec2(_114)), int(gl_LocalInvocationID.z))) + _162) + texelFetch(image_array_src, ivec3(uvec3(uvec2(_114), gl_LocalInvocationID.z)), int(gl_LocalInvocationID.z) + 1)) + texelFetch(image_array_src, ivec3(uvec3(uvec2(_114), uint(int(gl_LocalInvocationID.z)))), int(gl_LocalInvocationID.z) + 1));
}


//////////////////////////////////////
// Entry point: "depth_load" (comp) //
//////////////////////////////////////
#version 460
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

ivec2 naga_mod(ivec2 lhs, ivec2 rhs)
{
    bvec2 _79 = equal(rhs, ivec2(0));
    bvec2 _84 = equal(lhs, ivec2(int(0x80000000)));
    bvec2 _85 = equal(rhs, ivec2(-1));
    bvec2 _86 = bvec2(_84.x && _85.x, _84.y && _85.y);
    return lhs - mix(rhs, ivec2(1), bvec2(_79.x || _86.x, _79.y || _86.y)) * (lhs / mix(rhs, ivec2(1), bvec2(_79.x || _86.x, _79.y || _86.y)));
}

void main()
{
    ivec2 _206 = naga_mod(ivec2(uvec2(imageSize(image_storage_src)) * gl_LocalInvocationID.xy), ivec2(10, 20));
    imageStore(image_dst, _206.x, uvec4(uint(clamp(texelFetch(image_depth_multisampled_src, _206, int(gl_LocalInvocationID.z)).x, 0.0, 4294967040.0))));
}


///////////////////////////////////
// Entry point: "queries" (vert) //
///////////////////////////////////
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

void main()
{
    uint _230 = uint(textureSize(image_1d, int(0u)));
    gl_Position = vec4(float((((((((((_230 + uvec2(textureSize(image_2d, int(0u))).y) + uvec2(textureSize(image_2d, 1)).y) + uvec3(textureSize(image_2d_array, int(0u))).xy.y) + uvec3(textureSize(image_2d_array, 1)).xy.y) + uvec2(textureSize(image_cube, int(0u))).y) + uvec2(textureSize(image_cube, 1)).y) + uvec3(textureSize(image_cube_array, int(0u))).xx.y) + uvec3(textureSize(image_cube_array, 1)).xx.y) + uvec3(textureSize(image_3d, int(0u))).z) + uvec3(textureSize(image_3d, 1)).z));
}


//////////////////////////////////////////
// Entry point: "levels_queries" (vert) //
//////////////////////////////////////////
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

void main()
{
    gl_Position = vec4(float(((((((uvec3(textureSize(image_2d_array, int(0u))).z + uvec3(textureSize(image_cube_array, int(0u))).z) + uint(textureSamples(image_aa))) + uint(textureQueryLevels(image_2d))) + uint(textureQueryLevels(image_2d_array))) + uint(textureQueryLevels(image_3d))) + uint(textureQueryLevels(image_cube))) + uint(textureQueryLevels(image_cube_array))));
}


//////////////////////////////////////////
// Entry point: "texture_sample" (frag) //
//////////////////////////////////////////
#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_spirv_intrinsics : require

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

layout(location = 0) out vec4 _302;

spirv_instruction(set = "GLSL.std.450", id = 81) float spvNClamp(float, float, float);
spirv_instruction(set = "GLSL.std.450", id = 81) vec2 spvNClamp(vec2, vec2, vec2);
spirv_instruction(set = "GLSL.std.450", id = 81) vec3 spvNClamp(vec3, vec3, vec3);
spirv_instruction(set = "GLSL.std.450", id = 81) vec4 spvNClamp(vec4, vec4, vec4);

void main()
{
    vec4 a = vec4(0.0);
    a += texture(sampler1D(image_1d, sampler_reg), 0.5);
    a += texture(sampler2D(image_2d, sampler_reg), vec2(0.5));
    a += textureOffset(sampler2D(image_2d, sampler_reg), vec2(0.5), ivec2(3, 1));
    a += textureLod(sampler2D(image_2d, sampler_reg), vec2(0.5), 2.2999999523162841796875);
    a += textureLodOffset(sampler2D(image_2d, sampler_reg), vec2(0.5), 2.2999999523162841796875, ivec2(3, 1));
    a += textureOffset(sampler2D(image_2d, sampler_reg), vec2(0.5), ivec2(3, 1), 2.0);
    vec2 _351 = vec2(0.5) / vec2(uvec2(textureSize(image_2d, int(0u))));
    a += textureLod(sampler2D(image_2d, sampler_reg), spvNClamp(vec2(0.5), _351, vec2(1.0) - _351), 0.0);
    float _361 = float(0u);
    a += texture(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _361));
    float _367 = float(0u);
    a += textureOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _367), ivec2(3, 1));
    float _373 = float(0u);
    a += textureLod(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _373), 2.2999999523162841796875);
    float _379 = float(0u);
    a += textureLodOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _379), 2.2999999523162841796875, ivec2(3, 1));
    float _385 = float(0u);
    a += textureOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _385), ivec2(3, 1), 2.0);
    float _391 = float(0);
    a += texture(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _391));
    float _397 = float(0);
    a += textureOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _397), ivec2(3, 1));
    float _403 = float(0);
    a += textureLod(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _403), 2.2999999523162841796875);
    float _409 = float(0);
    a += textureLodOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _409), 2.2999999523162841796875, ivec2(3, 1));
    float _415 = float(0);
    a += textureOffset(sampler2DArray(image_2d_array, sampler_reg), vec3(vec2(0.5), _415), ivec2(3, 1), 2.0);
    float _422 = float(0u);
    a += texture(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _422));
    float _428 = float(0u);
    a += textureLod(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _428), 2.2999999523162841796875);
    float _434 = float(0u);
    a += texture(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _434), 2.0);
    float _440 = float(0);
    a += texture(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _440));
    float _446 = float(0);
    a += textureLod(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _446), 2.2999999523162841796875);
    float _452 = float(0);
    a += texture(samplerCubeArray(image_cube_array, sampler_reg), vec4(vec3(0.5), _452), 2.0);
    _302 = a;
}


/////////////////////////////////////////////////////
// Entry point: "texture_sample_comparison" (frag) //
/////////////////////////////////////////////////////
#version 460

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform samplerShadow sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

layout(location = 0) out float _459;

void main()
{
    float a = 0.0;
    a += texture(sampler2DShadow(image_2d_depth, sampler_cmp), vec3(vec2(0.5), 0.5));
    float _477 = float(0u);
    a += texture(sampler2DArrayShadow(image_2d_array_depth, sampler_cmp), vec4(vec3(vec2(0.5), _477), 0.5));
    float _483 = float(0);
    a += texture(sampler2DArrayShadow(image_2d_array_depth, sampler_cmp), vec4(vec3(vec2(0.5), _483), 0.5));
    a += texture(samplerCubeShadow(image_cube_depth, sampler_cmp), vec4(vec3(0.5), 0.5));
    a += textureLod(sampler2DShadow(image_2d_depth, sampler_cmp), vec3(vec2(0.5), 0.5), 0.0);
    float _498 = float(0u);
    a += textureGrad(sampler2DArrayShadow(image_2d_array_depth, sampler_cmp), vec4(vec3(vec2(0.5), _498), 0.5), vec2(0.0), vec2(0.0));
    float _504 = float(0);
    a += textureGrad(sampler2DArrayShadow(image_2d_array_depth, sampler_cmp), vec4(vec3(vec2(0.5), _504), 0.5), vec2(0.0), vec2(0.0));
    a += textureGrad(samplerCubeShadow(image_cube_depth, sampler_cmp), vec4(vec3(0.5), 0.5), vec3(0.0), vec3(0.0));
    _459 = a;
}


//////////////////////////////////
// Entry point: "gather" (frag) //
//////////////////////////////////
#version 460

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform samplerShadow sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

layout(location = 0) out vec4 _515;

void main()
{
    _515 = (((textureGather(sampler2D(image_2d, sampler_reg), vec2(0.5), int(1u)) + textureGatherOffset(sampler2D(image_2d, sampler_reg), vec2(0.5), ivec2(3, 1), int(3u))) + textureGather(sampler2DShadow(image_2d_depth, sampler_cmp), vec2(0.5), 0.5)) + textureGatherOffset(sampler2DShadow(image_2d_depth, sampler_cmp), vec2(0.5), 0.5, ivec2(3, 1))) + (vec4(textureGather(usampler2D(image_2d_u32, sampler_reg), vec2(0.5))) + vec4(textureGather(isampler2D(image_2d_i32, sampler_reg), vec2(0.5))));
}


///////////////////////////////////////////////
// Entry point: "depth_no_comparison" (frag) //
///////////////////////////////////////////////
#version 460

layout(set = 0, binding = 0) uniform utexture2D image_mipmapped_src;
layout(set = 0, binding = 3) uniform utexture2DMS image_multisampled_src;
layout(set = 0, binding = 4) uniform texture2DMS image_depth_multisampled_src;
layout(set = 0, binding = 1, rgba8ui) uniform readonly uimage2D image_storage_src;
layout(set = 0, binding = 5) uniform utexture2DArray image_array_src;
layout(set = 0, binding = 6, r32ui) uniform readonly uimage1D image_dup_src;
layout(set = 0, binding = 7) uniform utexture1D image_1d_src;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage1D image_dst;
layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform utexture2D image_2d_u32;
layout(set = 0, binding = 3) uniform itexture2D image_2d_i32;
layout(set = 0, binding = 4) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 5) uniform textureCube image_cube;
layout(set = 0, binding = 6) uniform textureCubeArray image_cube_array;
layout(set = 0, binding = 7) uniform texture3D image_3d;
layout(set = 0, binding = 8) uniform texture2DMS image_aa;
layout(set = 1, binding = 0) uniform sampler sampler_reg;
layout(set = 1, binding = 1) uniform sampler sampler_cmp;
layout(set = 1, binding = 2) uniform texture2D image_2d_depth;
layout(set = 1, binding = 3) uniform texture2DArray image_2d_array_depth;
layout(set = 1, binding = 4) uniform textureCube image_cube_depth;

layout(location = 0) out vec4 _549;

void main()
{
    float _561 = float(1);
    _549 = (vec4(texture(sampler2DShadow(image_2d_depth, sampler_reg), vec2(0.5)).x) + textureGather(sampler2DShadow(image_2d_depth, sampler_reg), vec2(0.5))) + vec4(textureLod(sampler2DShadow(image_2d_depth, sampler_reg), vec2(0.5), _561).x);
}

