pub mod framework;
pub mod utils;

pub mod boids;
pub mod bunnymark;
pub mod conservative_raster;
pub mod cube;
pub mod hello;
pub mod hello_compute;
pub mod hello_synchronization;
pub mod hello_triangle;
pub mod hello_windows;
pub mod hello_workgroups;
pub mod mipmap;
pub mod msaa_line;
pub mod render_to_texture;
pub mod repeated_compute;
pub mod shadow;
pub mod skybox;
pub mod srgb_blend;
pub mod stencil_triangles;
pub mod storage_texture;
pub mod texture_arrays;
pub mod timestamp_queries;
pub mod uniform_values;
pub mod water;

#[cfg(test)]
wgpu_test::gpu_test_main!();
