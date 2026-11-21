#version 310 es

precision highp float;
precision highp int;

vec4 global = vec4(0.0);

vec4 global_1 = vec4(0.0);

layout(location = 0) out vec4 _fs2p_location0;

void function() {
    vec2 phi_49_ = vec2(0.0);
    vec4 _e6 = global;
    if (false) {
        phi_49_ = vec2((_e6.x * 0.5), (_e6.y * 0.5));
    } else {
        phi_49_ = vec2((_e6.x * 0.25), (_e6.y * 0.25));
    }
    vec2 _e17 = phi_49_;
    global_1[0u] = _e17.x;
    global_1[1u] = _e17.y;
    return;
}

void main() {
    vec4 param = gl_FragCoord;
    global = param;
    function();
    vec4 _e3 = global_1;
    _fs2p_location0 = _e3;
    return;
}

