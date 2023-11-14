#version 450

void main() {
    dvec4 a = dvec4(1.0);
    dvec4 b = dvec4(2.0);
    dmat4 m = dmat4(a, b, a, b);
    int i = 5;

    dvec4 ceilOut = ceil(a);
    dvec4 roundOut = round(a);
    dvec4 floorOut = floor(a);
    dvec4 fractOut = fract(a);
    dvec4 truncOut = trunc(a);
    dvec4 absOut = abs(a);
    dvec4 sqrtOut = sqrt(a);
    dvec4 inversesqrtOut = inversesqrt(a);
    dvec4 signOut = sign(a);
    dmat4 transposeOut = transpose(m);
    dvec4 normalizeOut = normalize(a);
    double lengthOut = length(a);
    double determinantOut = determinant(m);
    double modOut = mod(a.x, b.x);
    double dotOut = dot(a, b);
    dvec4 maxOut = max(a, b);
    dvec4 minOut = min(a, b);
    dvec4 reflectOut = reflect(a, b);
    dvec3 crossOut = cross(a.xyz, b.xyz);
    double distanceOut = distance(a, b);
    dvec4 stepOut = step(a, b);
    double ldexpOut = ldexp(a.x, i);
    double smoothStepScalar = smoothstep(0.0, 1.0, 0.5);
    dvec4 smoothStepVector = smoothstep(dvec4(0.0), dvec4(1.0), dvec4(0.5));
    dvec4 smoothStepMixed = smoothstep(0.0, 1.0, dvec4(0.5));
}
