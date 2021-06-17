fn main() {
    var a: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    a.x = 2.0;
    return;
}

[[stage(vertex)]]
fn main1() {
    main();
    return;
}
