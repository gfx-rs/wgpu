#ifdef GL_ES
precision mediump float;
#endif

// Fallback: use attribute for older versions
#if __VERSION__ < 130
    attribute float a_vertexId;
    #define gl_VertexID int(a_vertexId)
#endif

void main() {
    vec2 pos;
    if (gl_VertexID == 0) {
        pos = vec2( 0.0,  0.5);
    } else if (gl_VertexID == 1) {
        pos = vec2(-0.5, -0.5);
    } else {
        pos = vec2( 0.5, -0.5);
    }
    
    gl_Position = vec4(pos, 0.0, 1.0);
}