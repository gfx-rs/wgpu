[[block]]
struct PushConstants {
    example: f32;
};

var<push_constant> c: PushConstants;

fn main() {
    return;
}

[[stage(vertex)]]
fn main1() {
    main();
    return;
}
