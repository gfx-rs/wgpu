// `@payload`` attribute. It is not possible for this attribute to reach the validator
// without the extension enabled, because the attribute is only allowed on mesh and task
// stages, and those stages are rejected (with or without the `@payload` attribute) when
// the mesh shader extension is not enabled.
//
// There is a direct-to-validator test case for `@payload` in `validation.rs`.

@compute @workgroup_size(1) @payload(foo)
        fn main() {}
