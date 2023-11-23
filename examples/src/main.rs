fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("please provide example name!");

    match &*arg1 {
        "boids" => wgpu_examples::boids::main(),
        "bunnymark" => wgpu_examples::bunnymark::main(),
        "conservative_raster" => wgpu_examples::conservative_raster::main(),
        "cube" => wgpu_examples::cube::main(),
        "hello" => wgpu_examples::hello::main(),
        "hello_compute" => wgpu_examples::hello_compute::main(),
        "hello_synchronization" => wgpu_examples::hello_synchronization::main(),
        "hello_triangle" => wgpu_examples::hello_triangle::main(),
        "hello_windows" => wgpu_examples::hello_windows::main(),
        "hello_workgroups" => wgpu_examples::hello_workgroups::main(),
        "mipmap" => wgpu_examples::mipmap::main(),
        "msaa_line" => wgpu_examples::msaa_line::main(),
        "render_to_texture" => wgpu_examples::render_to_texture::main(),
        "repeated_compute" => wgpu_examples::repeated_compute::main(),
        "shadow" => wgpu_examples::shadow::main(),
        "skybox" => wgpu_examples::skybox::main(),
        "srgb_blend" => wgpu_examples::srgb_blend::main(),
        "stencil_triangles" => wgpu_examples::stencil_triangles::main(),
        "storage_texture" => wgpu_examples::storage_texture::main(),
        "texture_arrays" => wgpu_examples::texture_arrays::main(),
        "timestamp_queries" => wgpu_examples::timestamp_queries::main(),
        "uniform_values" => wgpu_examples::uniform_values::main(),
        "water" => wgpu_examples::water::main(),
        e => panic!("unknown example: {}", e),
    }
}
