@group(0) @binding(0)
var<storage, read_write> scratch: u32;

@compute
@workgroup_size(1)
fn main() {
    scratch = 1u;
}

@group(0) @binding(0)
var<storage, read_write> vertices: array<f32>;

@compute @workgroup_size(1)
fn generate_vertices() {
    let triangle = array<f32, 9>(1.0, 1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    for (var i = 0u; i < 9u; i++) {
        vertices[i] = triangle[i];
    }
}
