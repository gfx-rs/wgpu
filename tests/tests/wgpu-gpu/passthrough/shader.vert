// Fallback: use attribute for older versions
#if __VERSION__ < 130
    attribute float a_vertexId;
    #define vertexId int(a_vertexId)
#else
    #define vertexId gl_VertexID
#endif

void main() {
    vec2 pos;
    if (vertexId == 0) {
        pos = vec2( 0.0,  0.5);
    } else if (vertexId == 1) {
        pos = vec2(-0.5, -0.5);
    } else {
        pos = vec2( 0.5, -0.5);
    }
    
    gl_Position = vec4(pos, 0.0, 1.0);
}