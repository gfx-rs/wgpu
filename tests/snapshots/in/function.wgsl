fn test_function(test: f32) -> f32 {
    return test;
}

[[stage(vertex)]]
fn main() -> void {
    var test: f32 = test_function(1.0);
}
