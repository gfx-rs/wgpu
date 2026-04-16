#if __VERSION__ >= 130
    out vec4 fragColor;
#else
    #define fragColor gl_FragColor
#endif


void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}