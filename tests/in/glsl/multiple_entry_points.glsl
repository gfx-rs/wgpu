#version 450
// vertex
void vert_main() {
  gl_Position = vec4(1.0, 1.0, 1.0, 1.0);
}

// fragment
layout(location = 0) out vec4 o_color;
void frag_main() {
  o_color = vec4(1.0, 1.0, 1.0, 1.0);
}

//compute
// layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void comp_main() {
    if (gl_GlobalInvocationID.x > 1) {
        return;
    }
}
