use wgpu_test::infra::GpuTest;

mod regression {
    mod issue_3457;
}
mod buffer;
mod buffer_copy;
mod buffer_usages;
mod clear_texture;
mod device;
mod encoder;
mod example_wgsl;
mod external_texture;
mod instance;
mod poll;
mod queue_transfer;
mod resource_descriptor_accessor;
mod resource_error;
mod shader;
mod shader_primitive_index;
mod shader_view_format;
mod texture_bounds;
mod transfer;
mod vertex_indices;
mod write_texture;
mod zero_init_texture_after_discard;

// wasm_bindgen_test_configure!(run_in_browser);

fn main() -> wgpu_test::infra::MainResult {
    wgpu_test::infra::main([
        instance::InitializeTest::new(),
        queue_transfer::QueueWriteTextureOverflowTest::new(),
        resource_descriptor_accessor::BufferSizeAndUsageTest::new(),
        resource_error::BadBufferTest::new(),
        resource_error::BadTextureTest::new(),
        texture_bounds::BadCopyOriginTest::new(),
        transfer::CopyOverflowZTest::new(),
        write_texture::WriteTextureSubset2dTest::new(),
        write_texture::WriteTextureSubset3dTest::new(),
        zero_init_texture_after_discard::DiscardingColorTargetResetsTextureInitStateCheckVisibleOnCopyAfterSubmitTest::new(),
        zero_init_texture_after_discard::DiscardingColorTargetResetsTextureInitStateCheckVisibleOnCopyInSameEncoderTest::new(),
        zero_init_texture_after_discard::DiscardingDepthTargetResetsTextureInitStateCheckVisibleOnCopyInSameEncoderTest::new(),
        zero_init_texture_after_discard::DiscardingEitherDepthOrStencilAspectTest::new(),
    ])
}
