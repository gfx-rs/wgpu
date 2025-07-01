const array_: array<f32, 2> = array<f32, 2>(1f, 2f);

fn main_1() {
    var local: array<f32, 2> = array_;

    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
