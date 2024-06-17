#version 310 es

precision highp float;
precision highp int;

uint global_2 = 0u;

vec4 global_3 = vec4(0.0, 0.0, 0.0, 1.0);

invariant gl_Position;

void function_1() {
    vec4 local[6] = vec4[6](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    uint _e5 = global_2;
    local = vec4[6](vec4(-1.0, -1.0, 0.0, 1.0), vec4(1.0, -1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), vec4(-1.0, 1.0, 0.0, 1.0), vec4(-1.0, -1.0, 0.0, 1.0));
    if ((_e5 < 6u)) {
        vec4 _e8 = local[_e5];
        global_3 = _e8;
    }
    return;
}

void main() {
    uint param_1 = uint(gl_VertexID);
    global_2 = param_1;
    function_1();
    float _e4 = global_3.y;
    global_3.y = -(_e4);
    vec4 _e6 = global_3;
    gl_Position = _e6;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

