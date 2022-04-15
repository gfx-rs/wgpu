struct PushConstants {
    example: f32,
}

var<push_constant> c: PushConstants;

fn main_1() {
    return;
}

@vertex 
fn main() {
    main_1();
    return;
}
