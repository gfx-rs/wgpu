#version 450

layout(early_fragment_tests) in;

layout(location = 0) in vec4 v_Colour;
// Comment this out if using user-clipping planes:
layout(location = 1) in float v_ClipDist;

layout(location = 0) out vec4 outColour;

void main() {
    // Comment this out if using user-clipping planes:
    if(v_ClipDist < 0.0) {
        discard;
    }

    outColour = v_Colour;
    outColour.a = 1.0;
}
