//! Tests that `Device::generate_allocator_report` accounts for the buffers and textures a backend
//! allocates, and stops accounting for them once they are destroyed.

use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(ALLOCATOR_REPORT_TRACKS_BUFFERS_AND_TEXTURES);
}

const BUFFER_SIZE: u64 = 4 * 1024 * 1024;
const TEXTURE_SIDE: u32 = 512;
/// `Rgba8Unorm`, one mip level, one layer.
const TEXTURE_SIZE: u64 = TEXTURE_SIDE as u64 * TEXTURE_SIDE as u64 * 4;

#[apply(gpu_test!)]
static ALLOCATOR_REPORT_TRACKS_BUFFERS_AND_TEXTURES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(|ctx| async move {
            // Vulkan, DX12 and Metal report their allocations; GL, WebGPU and noop have no report.
            let reports = matches!(
                ctx.adapter_info.backend,
                wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal
            );
            let Some(before) = ctx.device.generate_allocator_report() else {
                assert!(
                    !reports,
                    "the {:?} backend returned no allocator report",
                    ctx.adapter_info.backend
                );
                return;
            };

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("allocator report buffer"),
                size: BUFFER_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("allocator report texture"),
                size: wgpu::Extent3d {
                    width: TEXTURE_SIDE,
                    height: TEXTURE_SIDE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let during = ctx
                .device
                .generate_allocator_report()
                .expect("the backend produced a report before, so it should now");
            let grew = during
                .total_allocated_bytes
                .saturating_sub(before.total_allocated_bytes);
            assert!(
                grew >= BUFFER_SIZE + TEXTURE_SIZE,
                "a {BUFFER_SIZE} B buffer and a {TEXTURE_SIZE} B texture moved \
                 total_allocated_bytes by only {grew} B\nbefore: {before:?}\nduring: {during:?}",
            );
            assert!(
                during.total_reserved_bytes >= during.total_allocated_bytes,
                "reserved bytes are less than allocated bytes: {during:?}",
            );

            buffer.destroy();
            texture.destroy();
            drop(buffer);
            drop(texture);
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();

            let after = ctx
                .device
                .generate_allocator_report()
                .expect("the backend produced a report before, so it should now");
            let released = during
                .total_allocated_bytes
                .saturating_sub(after.total_allocated_bytes);
            assert!(
                released >= BUFFER_SIZE + TEXTURE_SIZE,
                "destroying the buffer and texture released only {released} B of \
                 total_allocated_bytes\nduring: {during:?}\nafter: {after:?}",
            );
        });
