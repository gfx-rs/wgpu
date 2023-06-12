const array_: array<f32, 2> = array<f32, 2>(1.0, 2.0);
var<private> i: u32;

fn main_1() {
    var local: array<f32, 2> = array<f32, 2>(1.0, 2.0);

    let _e2 = i;
}

@fragment 
fn main() {
    main_1();
    return;
}
