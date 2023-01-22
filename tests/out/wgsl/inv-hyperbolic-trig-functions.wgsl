var<private> a: f32;

fn main_1() {
    var b: f32;
    var c: f32;
    var d: f32;

    let _e8 = a;
    b = asinh(_e8);
    let _e10 = a;
    c = acosh(_e10);
    let _e12 = a;
    d = atanh(_e12);
    return;
}

@vertex 
fn main() {
    main_1();
}
