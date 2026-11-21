// The outer constructor has the correct number of components, but only
// because the inner constructor has too many.

fn x() {
    _ = vec4(1, vec2(2, 3, 4));
}
