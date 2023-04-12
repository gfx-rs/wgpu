#version 450

void ftest(vec4 a, vec4 b) {
	bvec4 c = lessThan(a, b);
	bvec4 d = lessThanEqual(a, b);
	bvec4 e = greaterThan(a, b);
	bvec4 f = greaterThanEqual(a, b);
	bvec4 g = equal(a, b);
	bvec4 h = notEqual(a, b);
}

void dtest(dvec4 a, dvec4 b) {
	bvec4 c = lessThan(a, b);
	bvec4 d = lessThanEqual(a, b);
	bvec4 e = greaterThan(a, b);
	bvec4 f = greaterThanEqual(a, b);
	bvec4 g = equal(a, b);
	bvec4 h = notEqual(a, b);
}

void itest(ivec4 a, ivec4 b) {
	bvec4 c = lessThan(a, b);
	bvec4 d = lessThanEqual(a, b);
	bvec4 e = greaterThan(a, b);
	bvec4 f = greaterThanEqual(a, b);
	bvec4 g = equal(a, b);
	bvec4 h = notEqual(a, b);
}

void utest(uvec4 a, uvec4 b) {
	bvec4 c = lessThan(a, b);
	bvec4 d = lessThanEqual(a, b);
	bvec4 e = greaterThan(a, b);
	bvec4 f = greaterThanEqual(a, b);
	bvec4 g = equal(a, b);
	bvec4 h = notEqual(a, b);
}

void btest(bvec4 a, bvec4 b) {
	bvec4 c = equal(a, b);
	bvec4 d = notEqual(a, b);
	bool e = any(a);
	bool f = all(a);
	bvec4 g = not(a);
}

void main() {}
