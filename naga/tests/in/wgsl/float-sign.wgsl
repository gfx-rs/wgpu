// Regression test for `sign` lowering in the HLSL backend
// (https://bugzilla.mozilla.org/show_bug.cgi?id=2011051). HLSL's `sign`
// intrinsic returns a signed integer even for floating-point arguments, so the
// result must be converted back to the argument's type. Storing an unconverted
// result reinterprets the integer's bits, turning `sign(-1.0)` into a NaN.

@group(0) @binding(0) var<storage, read_write> floats: array<f32, 4>;
@group(0) @binding(1) var<storage, read_write> ints: array<i32, 4>;

@compute @workgroup_size(1)
fn main() {
    floats[0] = sign(floats[1]);

    let v = vec2(floats[2], floats[3]);
    let signs = sign(v);
    floats[2] = signs.x;
    floats[3] = signs.y;

    // Unlike the floating-point overloads, HLSL's `sign` already returns the
    // right type here, so no conversion is emitted.
    ints[0] = sign(ints[1]);
}
