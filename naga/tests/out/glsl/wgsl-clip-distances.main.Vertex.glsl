#version 330 core
struct VertexOutput {
    vec4 position;
    float clip_distances[1];
};
out float gl_ClipDistance[1];

void main() {
    VertexOutput out_ = VertexOutput(vec4(0.0), float[1](0.0));
    out_.clip_distances[0] = 0.5;
    VertexOutput _e4 = out_;
    gl_Position = _e4.position;
    gl_ClipDistance = _e4.clip_distances;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

