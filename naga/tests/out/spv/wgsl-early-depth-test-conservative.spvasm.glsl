#version 460
layout(depth_less) out float gl_FragDepth;

void main()
{
    gl_FragDepth = gl_FragCoord.z - 0.100000001490116119384765625;
}

