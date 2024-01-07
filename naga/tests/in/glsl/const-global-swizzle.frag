// ISSUE: #4773
#version 450

#define MIX2(c) c.xy

layout(location = 0) in vec2 v_Uv;

layout(location = 0) out vec4 o_Target;

const vec2 blank = MIX2(vec2(0.0, 1.0));

void main() {
    vec2 col = MIX2(v_Uv) * blank;
    o_Target = vec4(col, 0.0, 1.0);
}