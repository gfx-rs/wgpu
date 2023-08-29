// Standard functions.

fn test_any_and_all_for_bool() -> bool {
    let a = any(true);
    return all(a);
}


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

    let a = test_any_and_all_for_bool();

    return (x + y) * z;
}
