import "GLSL.std.450" as std::glsl;

fn test_function(test: f32) -> f32 {
    return test;
}

fn main_vert() -> void {
    var foo: f32 = std::glsl::distance(0.0, 1.0);
    var test: f32 = test_function(1.0);
}

entry_point vertex as "main" = main_vert;