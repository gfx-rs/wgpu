#version 450

const vec3 water_colour = vec3(0.0, 117.0 / 255.0, 242.0 / 255.0);
const float zNear = 10.0;
const float zFar = 400.0;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4x4 _view;
    mat4x4 _projection;
    vec4 time_size_width;
    float viewport_height;
};

layout(set = 0, binding = 1) uniform texture2D reflection;
layout(set = 0, binding = 2) uniform texture2D terrain_depth_tex;
layout(set = 0, binding = 3) uniform sampler colour_sampler;

layout(location = 0) in vec2 f_WaterScreenPos;
layout(location = 1) in float f_Fresnel;
layout(location = 2) in vec3 f_Light;

layout(location = 0) out vec4 outColor;

float to_linear_depth(float depth) {
    float z_n = 2.0 * depth - 1.0;
    float z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
    return z_e;
}

void main() {
    vec3 reflection_colour = texture(sampler2D(reflection, colour_sampler), f_WaterScreenPos.xy).xyz;

    float pixel_depth = to_linear_depth(gl_FragCoord.z);
    float terrain_depth = to_linear_depth(texture(sampler2D(terrain_depth_tex, colour_sampler), gl_FragCoord.xy / vec2(time_size_width.w, viewport_height)).r);

    float dist = terrain_depth - pixel_depth;
    float clamped = pow(smoothstep(0.0, 1.5, dist), 4.8);

    outColor.a = clamped * (1.0 - f_Fresnel);

    vec3 final_colour = f_Light + reflection_colour;

    vec3 depth_colour = mix(final_colour, water_colour, smoothstep(1.0, 5.0, dist) * 0.2);

    outColor.xyz = depth_colour;
}
