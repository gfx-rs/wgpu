fn function1() -> f32 {
    return 0.0;
}

fn main1() {
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}
