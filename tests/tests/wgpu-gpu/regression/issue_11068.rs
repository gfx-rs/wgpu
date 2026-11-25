use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(UNALIGNED_BUFFER_TEXTURE_COPY_MEMORY_BALANCED);
}

/// A buffer-to-texture copy at an offset the device cannot copy from directly
/// goes through an intermediate buffer on DX12; once the copies are done, the
/// buffer memory has to be back where it was.
#[apply(gpu_test!)]
static UNALIGNED_BUFFER_TEXTURE_COPY_MEMORY_BALANCED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // NOTE: The WebGPU backend does not implement internal counters.
                .skip(FailureCase::backend(wgpu::Backends::BROWSER_WEBGPU)),
        )
        .run_async(|ctx| async move {
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 4,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Uint,
                usage: wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
                texture_binding_view_dimension: None,
            });
            let copy = || {
                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 2,
                            bytes_per_row: None,
                            rows_per_image: None,
                        },
                    },
                    texture.as_image_copy(),
                    wgpu::Extent3d {
                        width: 4,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                ctx.queue.submit([encoder.finish()]);
            };

            // The first copy can make allocations that wgpu keeps, so measure
            // from after it.
            copy();
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
            let before = ctx.device.get_internal_counters().hal.buffer_memory.read();

            for _ in 0..10 {
                copy();
            }
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();

            let after = ctx.device.get_internal_counters().hal.buffer_memory.read();
            assert_eq!(
                after, before,
                "buffer memory should return to its baseline once the copies are done",
            );
        });
