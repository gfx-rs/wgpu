#version 450
layout(location = 0) in vec2 a_particlePos;
layout(location = 1) in vec2 a_particleVel;
layout(location = 2) in vec2 a_pos;

void main() {
    float angle = -atan(a_particleVel.x, a_particleVel.y);
    vec2 pos = vec2(a_pos.x * cos(angle) - a_pos.y * sin(angle),
                    a_pos.x * sin(angle) + a_pos.y * cos(angle));
    gl_Position = vec4(pos + a_particlePos, 0, 1);
}
