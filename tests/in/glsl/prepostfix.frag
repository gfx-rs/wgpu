#version 450 core

void main() {
	int scalar_target;
	int scalar = 1;
	scalar_target = scalar++;
	scalar_target = --scalar;

	uvec2 vec_target;
	uvec2 vec = uvec2(1);
	vec_target = vec--;
	vec_target = ++vec;

	mat4x3 mat_target;
	mat4x3 mat = mat4x3(1);
	mat_target = mat++;
	mat_target = --mat;
}
