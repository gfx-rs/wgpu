//! Tests that evaluating `wgpu::SomeType: Send + Sync` is cheap.
//!
//! This test must be located within the `wgpu` package so that it can use `cfg(send_sync)`.
//! It is a separate test target so that, if it fails to compile, it doesn’t block other builds.
//!
//! The implementations that supply the property this test is testing for are located in
//! `../src/dispatch.rs`, in the macro `explicit_send_sync_impl`.

#![cfg(send_sync)]
#![allow(clippy::type_complexity, clippy::explicit_auto_deref)]

/// 50 `Box`es that have to be looked through to evaluate auto traits.
/// 
/// This count is chosen so that if `wgpu` uses a significant amount of the default
/// recursion limit, this code will fail to compile.
#[rustfmt::skip]
struct Tower<T: ?Sized>(
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
    Box<Box<Box<Box<Box<Box<Box<Box<Box<Box<
        T
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
    >>>>>>>>>>
);

impl<T: ?Sized> core::ops::Deref for Tower<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // We could let auto-deref handle this, but where’s the fun in that --
        // er, I mean, rust-analyzer wouldn’t type check it properly.
        &**************************************************self.0
    }
}

const fn assert_send_sync_is_cheap<T>()
where
    Tower<T>: Send,
    Tower<T>: Sync,
{
}

const _: () = {
    assert_send_sync_is_cheap::<wgpu::Adapter>();
    assert_send_sync_is_cheap::<wgpu::BindGroup>();
    assert_send_sync_is_cheap::<wgpu::BindGroupLayout>();
    assert_send_sync_is_cheap::<wgpu::Blas>();
    assert_send_sync_is_cheap::<wgpu::Buffer>();
    assert_send_sync_is_cheap::<wgpu::BufferView>();
    assert_send_sync_is_cheap::<wgpu::BufferViewMut>();
    assert_send_sync_is_cheap::<wgpu::CommandBuffer>();
    assert_send_sync_is_cheap::<wgpu::CommandEncoder>();
    assert_send_sync_is_cheap::<wgpu::CompilationInfo>();
    assert_send_sync_is_cheap::<wgpu::ComputePass>();
    assert_send_sync_is_cheap::<wgpu::ComputePipeline>();
    assert_send_sync_is_cheap::<wgpu::Device>();
    assert_send_sync_is_cheap::<wgpu::Error>();
    assert_send_sync_is_cheap::<wgpu::ExternalTexture>();
    assert_send_sync_is_cheap::<wgpu::Features>();
    assert_send_sync_is_cheap::<wgpu::Instance>();
    assert_send_sync_is_cheap::<wgpu::Limits>();
    assert_send_sync_is_cheap::<wgpu::PipelineCache>();
    assert_send_sync_is_cheap::<wgpu::PipelineLayout>();
    assert_send_sync_is_cheap::<wgpu::QuerySet>();
    assert_send_sync_is_cheap::<wgpu::Queue>();
    assert_send_sync_is_cheap::<wgpu::QueueWriteBufferView>();
    assert_send_sync_is_cheap::<wgpu::RenderBundle>();
    // RenderBundleEncoder is never Send or Sync
    assert_send_sync_is_cheap::<wgpu::RenderPass>();
    assert_send_sync_is_cheap::<wgpu::RenderPipeline>();
    assert_send_sync_is_cheap::<wgpu::Sampler>();
    assert_send_sync_is_cheap::<wgpu::ShaderModule>();
    assert_send_sync_is_cheap::<wgpu::Surface>();
    assert_send_sync_is_cheap::<wgpu::SurfaceTexture>();
    assert_send_sync_is_cheap::<wgpu::Texture>();
    assert_send_sync_is_cheap::<wgpu::TextureView>();
    assert_send_sync_is_cheap::<wgpu::Tlas>();
};
