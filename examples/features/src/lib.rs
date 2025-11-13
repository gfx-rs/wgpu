#![allow(clippy::arc_with_non_send_sync)] // False positive on wasm
#![warn(clippy::allow_attributes)]

pub mod framework;
pub mod utils;

pub mod big_compute_buffers;
pub mod boids;
pub mod bunnymark;
pub mod conservative_raster;
pub mod cube;
pub mod hello_synchronization;
pub mod hello_triangle;
pub mod hello_windows;
pub mod hello_workgroups;
pub mod mesh_shader;
pub mod mipmap;
pub mod msaa_line;
pub mod multiple_render_targets;
pub mod ray_cube_compute;
pub mod ray_cube_fragment;
pub mod ray_cube_normals;
pub mod ray_scene;
pub mod ray_shadows;
pub mod ray_traced_triangle;
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
fn all_tests() -> Vec<wgpu_test::GpuTestInitializer> {
    #[cfg_attr(
        target_arch = "wasm32",
        expect(unused_mut, reason = "non-wasm32 needs this mutable")
    )]
    let mut test_list = vec![
        boids::TEST,
        bunnymark::TEST,
        conservative_raster::TEST,
        cube::TEST,
        cube::TEST_LINES,
        hello_synchronization::tests::SYNC,
        mipmap::TEST,
        mipmap::TEST_QUERY,
        msaa_line::TEST,
        multiple_render_targets::TEST,
        ray_cube_compute::TEST,
        ray_cube_fragment::TEST,
        ray_cube_normals::TEST,
        ray_scene::TEST,
        ray_shadows::TEST,
        ray_traced_triangle::TEST,
        shadow::TEST,
        skybox::TEST,
        skybox::TEST_ASTC,
        skybox::TEST_BCN,
        skybox::TEST_ETC2,
        srgb_blend::TEST_LINEAR,
        srgb_blend::TEST_SRGB,
        stencil_triangles::TEST,
        texture_arrays::TEST,
        texture_arrays::TEST_NON_UNIFORM,
        texture_arrays::TEST_UNIFORM,
        timestamp_queries::tests::TIMESTAMPS_ENCODER,
        timestamp_queries::tests::TIMESTAMPS_PASSES,
        timestamp_queries::tests::TIMESTAMPS_PASS_BOUNDARIES,
        water::TEST,
    ];

    #[cfg(not(target_arch = "wasm32"))]
    {
        test_list.push(big_compute_buffers::tests::TWO_BUFFERS);
    }

    test_list
}

#[cfg(test)]
wgpu_test::gpu_test_main!(all_tests());
