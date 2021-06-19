fn exact(a: f32) {
    var a1: f32;

    a1 = a;
    return;
}

fn exact1(a2: i32) {
    var a3: i32;

    a3 = a2;
    return;
}

fn implicit(a4: f32) {
    var a5: f32;

    a5 = a4;
    return;
}

fn implicit1(a6: i32) {
    var a7: i32;

    a7 = a6;
    return;
}

fn main1() {
    exact1(1);
    implicit(f32(1u));
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}
