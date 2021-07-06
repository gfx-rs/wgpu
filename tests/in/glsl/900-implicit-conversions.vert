// ISSUE: #900
#version 450

// Signature match call the second overload
void exact(float a) {}
void exact(int a) {}

// No signature match but one overload satisfies the cast rules
void implicit(float a) {}
void implicit(int a) {}

// All satisfy the kind condition but they have different dimensions
void implicit_dims(float v) {  }
void implicit_dims(vec2 v) {  }
void implicit_dims(vec3 v) {  }
void implicit_dims(vec4 v) {  }

void main() {
  exact(1);
  implicit(1u);
  implicit_dims(ivec3(1));
}
