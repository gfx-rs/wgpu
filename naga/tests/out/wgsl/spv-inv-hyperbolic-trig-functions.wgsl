var<private> a: f32;

fn main_1() {
    var b: f32;
    var c: f32;
    var d: f32;

    let _e4 = a;
    b = asinh(_e4);
    let _e6 = a;
    c = acosh(_e6);
    let _e8 = a;
    d = atanh(_e8);
    return;
}

@fragment 
fn main() {
    main_1();
}
