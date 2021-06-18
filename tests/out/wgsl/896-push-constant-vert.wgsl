[[block]]
struct PushConstants {
    example: f32;
};

var<push_constant> c: PushConstants;

fn main1() {
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}
