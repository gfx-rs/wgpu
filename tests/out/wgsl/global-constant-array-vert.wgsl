var<private> i: u32;

fn main_1() {
    var local: array<f32,2u> = array<f32,2u>(1.0, 2.0);

    let _e2 = i;
}

[[stage(vertex)]]
fn main() {
    main_1();
    return;
}
