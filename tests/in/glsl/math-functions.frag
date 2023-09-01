#version 450

void main() {
    vec4 a = vec4(1.0);
    vec4 b = vec4(2.0);
    mat4 m = mat4(a, b, a, b);
    int i = 5;

    vec4 ceilOut = ceil(a);
    vec4 roundOut = round(a);
    vec4 floorOut = floor(a);
    vec4 fractOut = fract(a);
    vec4 truncOut = trunc(a);
    vec4 sinOut = sin(a);
    vec4 absOut = abs(a);
    vec4 sqrtOut = sqrt(a);
    vec4 inversesqrtOut = inversesqrt(a);
    vec4 expOut = exp(a);
    vec4 exp2Out = exp2(a);
    vec4 signOut = sign(a);
    mat4 transposeOut = transpose(m);
    // TODO: support inverse function in wgsl output
    // mat4 inverseOut = inverse(m);
    vec4 normalizeOut = normalize(a);
    vec4 sinhOut = sinh(a);
    vec4 cosOut = cos(a);
    vec4 coshOut = cosh(a);
    vec4 tanOut = tan(a);
    vec4 tanhOut = tanh(a);
    vec4 acosOut = acos(a);
    vec4 asinOut = asin(a);
    vec4 logOut = log(a);
    vec4 log2Out = log2(a);
    float lengthOut = length(a);
    float determinantOut = determinant(m);
    int bitCountOut = bitCount(i);
    int bitfieldReverseOut = bitfieldReverse(i);
    float atanOut = atan(a.x);
    float atan2Out = atan(a.x, a.y);
    float modOut = mod(a.x, b.x);
    vec4 powOut = pow(a, b);
    float dotOut = dot(a, b);
    vec4 maxOut = max(a, b);
    vec4 minOut = min(a, b);
    vec4 reflectOut = reflect(a, b);
    vec3 crossOut = cross(a.xyz, b.xyz);
    mat4 outerProductOut = outerProduct(a, b);
    float distanceOut = distance(a, b);
    vec4 stepOut = step(a, b);
    // TODO: support out params in wgsl output
    // vec4 modfOut = modf(a, b);
    // vec4 frexpOut = frexp(a, b);
    float ldexpOut = ldexp(a.x, i);
    vec4 rad = radians(a);
    float deg = degrees(a.x);
    float smoothStepScalar = smoothstep(0.0, 1.0, 0.5);
    vec4 smoothStepVector = smoothstep(vec4(0.0), vec4(1.0), vec4(0.5));
    vec4 smoothStepMixed = smoothstep(0.0, 1.0, vec4(0.5));
}
