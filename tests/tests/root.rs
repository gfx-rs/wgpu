use wgpu_test::infra::GpuTest;

mod regression {
    pub mod issue_3457;
}
mod buffer;
mod buffer_copy;
mod buffer_usages;
mod clear_texture;
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
    wgpu_test::infra::main(
        [
            buffer_copy::CopyAlignmentTest::new(),
            buffer_usages::BufferUsageMappablePrimaryTest::new(),
            buffer_usages::BufferUsageTest::new(),
            buffer::EmptyBufferTest::new(),
            buffer::MapOffsetTest::new(),
            clear_texture::ClearTextureAstcTest::new(),
            clear_texture::ClearTextureBcTest::new(),
            clear_texture::ClearTextureD32S8Test::new(),
            clear_texture::ClearTextureDepthCompatTest::new(),
            clear_texture::ClearTextureEtc2Test::new(),
            clear_texture::ClearTextureUncompressedCompatTest::new(),
            clear_texture::ClearTextureUncompressedGlesCompatTest::new(),
            encoder::DropEncoderTest::new(),
            instance::InitializeTest::new(),
            poll::DoubleWaitOnSubmissionTest::new(),
            poll::DoubleWaitTest::new(),
            poll::WaitOnSubmissionTest::new(),
            poll::WaitOutOfOrderTest::new(),
            poll::WaitTest::new(),
            queue_transfer::QueueWriteTextureOverflowTest::new(),
            regression::issue_3457::PassResetVertexBufferTest::new(),
            resource_descriptor_accessor::BufferSizeAndUsageTest::new(),
            resource_error::BadBufferTest::new(),
            resource_error::BadTextureTest::new(),
            shader_primitive_index::DrawIndexedTest::new(),
            shader_primitive_index::DrawTest::new(),
            shader_view_format::ReinterpretSrgbTest::new(),
            shader::numeric_builtins::NumericBuiltinsTest::new(),
            shader::struct_layout::PushConstantInputTest::new(),
            shader::struct_layout::StorageInputTest::new(),
            shader::struct_layout::UniformInputTest::new(),
            shader::zero_init_workgroup_mem::ZeroInitWorkgroupMemTest::new(),
            texture_bounds::BadCopyOriginTest::new(),
            transfer::CopyOverflowZTest::new(),
            vertex_indices::DrawInstancedOffsetTest::new(),
            vertex_indices::DrawInstancedTest::new(),
            vertex_indices::DrawTest::new(),
            vertex_indices::DrawVertexTest::new(),
            write_texture::WriteTextureSubset2dTest::new(),
            write_texture::WriteTextureSubset3dTest::new(),
            zero_init_texture_after_discard::DiscardingColorTargetResetsTextureInitStateCheckVisibleOnCopyAfterSubmitTest::new(),
            zero_init_texture_after_discard::DiscardingColorTargetResetsTextureInitStateCheckVisibleOnCopyInSameEncoderTest::new(),
            zero_init_texture_after_discard::DiscardingDepthTargetResetsTextureInitStateCheckVisibleOnCopyInSameEncoderTest::new(),
            zero_init_texture_after_discard::DiscardingEitherDepthOrStencilAspectTest::new(),
        ], [
            wgpu_test::infra::cpu_test(example_wgsl::parse_example_wgsl)
        ]
    )
}
