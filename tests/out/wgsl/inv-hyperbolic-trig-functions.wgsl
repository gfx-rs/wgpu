var<private> a: f32;

fn main1() {
    var b: f32;
    var c: f32;
    var d: f32;

    let _e8: f32 = a;
    b = log(_e8 + sqrt(_e8 * _e8 + 1.0));
    let _e10: f32 = a;
    c = log(_e10 + sqrt(_e10 * _e10 - 1.0));
    let _e12: f32 = a;
    d = 0.5 * log((1.0 + _e12) / (1.0 - _e12));
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
}
