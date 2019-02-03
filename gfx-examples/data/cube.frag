#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target;
layout(set = 0, binding = 0) uniform texture2D t_Color;
layout(set = 0, binding = 1) uniform sampler s_Color;

void main() {
    vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
    float blend = dot(v_TexCoord-vec2(0.5,0.5), v_TexCoord-vec2(0.5,0.5));
    o_Target = mix(tex, vec4(0.0,0.0,0.0,0.0), blend*1.0);
}
