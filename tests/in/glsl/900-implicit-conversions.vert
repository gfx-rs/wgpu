// ISSUE: #900
#version 450

// Signature match call the second overload
void exact(float a) {}
void exact(int a) {}

// No signature match but one overload satisfies the cast rules
void implicit(float a) {}
void implicit(int a) {}

void main() {
  exact(1);
  implicit(1u);
}
