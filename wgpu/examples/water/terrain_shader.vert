#version 450

layout(set = 0, binding = 0) uniform Uniforms {
    mat4x4 projection_view;
    vec4 clipping_plane;
};

const vec3 light = vec3(150.0, 70.0, 0.0);
const vec3 light_colour = vec3(1.0, 250.0 / 255.0, 209.0 / 255.0);

const float ambient = 0.2;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 colour;

layout(location = 0) out vec4 v_Colour;
// Comment this out if using user-clipping planes:
layout(location = 1) out float v_ClipDist;

void main() {
    gl_Position = projection_view * vec4(position, 1.0);

    // https://www.desmos.com/calculator/nqgyaf8uvo

    vec3 normalized_light_direction = normalize(position - light);

    float brightness_diffuse = clamp(dot(normalized_light_direction, normal), 0.2, 1.0);

    v_Colour.rgb = max((brightness_diffuse + ambient) * light_colour * colour.rgb, 0.0);
    v_Colour.a = colour.a;

    // Comment this out if using user-clipping planes:
    v_ClipDist = dot(vec4(position, 1.0), clipping_plane);

    // Uncomment this if using user-clipping planes:
    // gl_ClipDistance[0] = dot(vec4(position, 1.0), clipping_plane);
}
