// NOTE: invalid combinations are tested in the `validation::bad_cross_builtin_args` test.
@compute @workgroup_size(1) fn main() {
    let a = cross(vec3(1., 0., 0.), vec3(0., 1., 0.));
}
