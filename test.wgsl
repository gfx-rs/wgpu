@group(0) @binding(0)
var<uniform> test: array<vec3f, 3>;

fn get_thing() -> array<vec3f, 3> {
    return test;
}

@workgroup_size(1)
@compute
fn main() {
    let hi = get_thing();
}