var<private> i: u32;

fn main1() {
    var local: array<f32,2> = array<f32,2>(1.0, 2.0);

    let e2: u32 = i;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}
