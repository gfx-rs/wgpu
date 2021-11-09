var<private> a: f32;

fn main_1() {
    var b: f32;
    var c: f32;
    var d: f32;

    let e8: f32 = a;
    b = log(e8 + sqrt(e8 * e8 + 1.0));
    let e10: f32 = a;
    c = log(e10 + sqrt(e10 * e10 - 1.0));
    let e12: f32 = a;
    d = 0.5 * log((1.0 + e12) / (1.0 - e12));
    return;
}

[[stage(vertex)]]
fn main() {
    main_1();
}
