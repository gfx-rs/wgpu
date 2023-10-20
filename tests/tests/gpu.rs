mod regression {
    mod issue_3457;
    mod issue_4024;
    mod issue_4122;
}

mod bgra8unorm_storage;
mod bind_group_layout_dedup;
mod buffer;
mod buffer_copy;
mod buffer_usages;
mod clear_texture;
mod create_surface_error;
mod device;
mod encoder;
mod external_texture;
mod instance;
mod occlusion_query;
mod partially_bounded_arrays;
mod pipeline;
mod poll;
mod query_set;
mod queue_transfer;
mod resource_descriptor_accessor;
mod resource_error;
mod scissor_tests;
mod shader;
mod shader_primitive_index;
mod shader_view_format;
mod texture_bounds;
mod transfer;
mod vertex_indices;
mod write_texture;
mod zero_init_texture_after_discard;

wgpu_test::gpu_test_main!();
