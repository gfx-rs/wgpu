// Standard functions.

@fragment
fn derivatives(@builtin(position) foo: vec4<f32>) -> @location(0) vec4<f32> {
    var x = dpdxCoarse(foo);
    var y = dpdyCoarse(foo);
    var z = fwidthCoarse(foo);

    x = dpdxFine(foo);
    y = dpdyFine(foo);
    z = fwidthFine(foo);

    x = dpdx(foo);
    y = dpdy(foo);
    z = fwidth(foo);

    return (x + y) * z;
}
