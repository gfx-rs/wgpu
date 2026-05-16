#version 450 core

// Exercises calling `texture()` directly on combined image-sampler uniforms
// (`sampler2D`, `sampler2DShadow`).  Naga synthesises a paired implicit sampler
// for each such uniform so the call can be lowered to `textureSample` /
// `textureSampleCompare` without requiring the explicit constructor syntax
// `texture(sampler2D(u_tex, u_samp), uv)`.

layout(set = 0, binding = 0) uniform sampler2D u_tex;
layout(set = 0, binding = 1) uniform sampler2DShadow u_shadow;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_shadow_coord;
layout(location = 0) out vec4 o_color;
layout(location = 1) out float o_shadow;

void main() {
    o_color = texture(u_tex, v_uv);
    o_shadow = texture(u_shadow, v_shadow_coord);
}
