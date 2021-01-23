#version 450 core

#define MAIN void main() {
#define V_POSITION layout(location=0) in vec4 a_position;
#define ASSIGN_POSITION gl_Position = a_position;

V_POSITION
        
MAIN
    ASSIGN_POSITION
}