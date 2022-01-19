fn function_() -> f32 {
    return 0.0;
}

fn main_1() {
    return;
}

@stage(vertex) 
fn main() {
    main_1();
    return;
}
