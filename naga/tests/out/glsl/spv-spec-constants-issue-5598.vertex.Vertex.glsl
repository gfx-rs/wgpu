#version 310 es

precision highp float;
precision highp int;

uint global = 0u;

vec4 global_1 = vec4(0.0, 0.0, 0.0, 1.0);

invariant gl_Position;

void function() {
    vec4 local[6] = vec4[6](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    uint _e5 = global;
    local = vec4[6](vec4(-1.0, -1.0, 0.0, 1.0), vec4(1.0, -1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), vec4(-1.0, 1.0, 0.0, 1.0), vec4(-1.0, -1.0, 0.0, 1.0));
    if ((_e5 < 6u)) {
        vec4 _e8 = local[_e5];
        global_1 = _e8;
    }
    return;
}

void main() {
    uint param = uint(gl_VertexID);
    global = param;
    function();
    float _e4 = global_1.y;
    global_1.y = -(_e4);
    vec4 _e6 = global_1;
    gl_Position = _e6;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

