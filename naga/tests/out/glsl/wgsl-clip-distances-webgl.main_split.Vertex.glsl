#version 300 es
#extension GL_ANGLE_clip_cull_distance : require

precision highp float;
precision highp int;

struct VertexOutput {
    vec4 position;
    float clip_distances[1];
};
struct VertexOutputSplit {
    vec4 color;
    float clip_distances[6];
    vec4 position;
};
smooth out vec4 _vs2fs_location0;
out float gl_ClipDistance[6];

void main() {
    VertexOutputSplit out_1 = VertexOutputSplit(vec4(0.0), float[6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0), vec4(0.0));
    int i = 0;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            int _e14 = i;
            i = (_e14 + 1);
        }
        loop_init = false;
        int _e3 = i;
        if ((_e3 < 6)) {
        } else {
            break;
        }
        {
            int _e7 = i;
            int _e9 = i;
            out_1.clip_distances[_e7] = (float(_e9) - 2.5);
        }
    }
    VertexOutputSplit _e16 = out_1;
    _vs2fs_location0 = _e16.color;
    gl_ClipDistance = _e16.clip_distances;
    gl_Position = _e16.position;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

