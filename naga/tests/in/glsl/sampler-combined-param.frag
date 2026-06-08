#version 450 core

// Exercises passing combined image-sampler types as function parameters.
// Previously, naga's GLSL frontend would emit "Bad call" when texture()
// was called through a sampler2D function parameter because no companion
// sampler was synthesised for the parameter.

layout(set = 0, binding = 0) uniform sampler2D u_lightmap;
layout(set = 0, binding = 1) uniform sampler2DShadow u_shadow;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_shadow_coord;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_shadow;

// Basic: combined sampler passed as a function parameter
vec4 sample_tex(sampler2D tex, vec2 uv) {
    return texture(tex, uv);
}

// Shadow sampler as a function parameter
float sample_shadow(sampler2DShadow sm, vec3 coord) {
    return texture(sm, coord);
}

// Forwarding a combined sampler from one user function to another
vec4 inner(sampler2D tex, vec2 uv) {
    return texture(tex, uv);
}

vec4 outer(sampler2D tex, vec2 uv) {
    return inner(tex, uv);
}

void main() {
    o_color = sample_tex(u_lightmap, v_uv);
    o_shadow = sample_shadow(u_shadow, v_shadow_coord);
    o_color += outer(u_lightmap, v_uv);
}
