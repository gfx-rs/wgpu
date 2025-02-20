const constant: i32 = 10i;

fn function() -> f32 {
    return 0f;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
