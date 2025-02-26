// NOTE: invalid combinations are tested in the `validation::bad_cross_builtin_args` test.
@compute @workgroup_size(1) fn main() {
    let a = cross(vec3(0., 1., 2.), vec3(0., 1., 2.));
}
