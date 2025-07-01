use wgpu::{util::DeviceExt, Backends};
use wgpu_test::{fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static BUFFER_DESTROY: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        buffer.destroy();

        buffer.destroy();

        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

        fail(
            &ctx.device,
            || {
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Write, move |_| {});
            },
            Some("buffer with 'buffer' label has been destroyed"),
        );

        buffer.destroy();

        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

        buffer.destroy();

        buffer.destroy();

        let descriptor = wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };

        // Scopes to mix up the drop/poll ordering.
        {
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
        }
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        {
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            ctx.async_poll(wgpu::PollType::wait()).await.unwrap();
            buffer.destroy();
        }
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();
    });

#[gpu_test]
static TEXTURE_DESTROY: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // multisampling is not supported for clear
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        texture.destroy();

        texture.destroy();

        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

        texture.destroy();

        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

        texture.destroy();

        texture.destroy();
    });

// Test that destroying a buffer between command buffer recording and
// submission fails gracefully.
#[gpu_test]
static BUFFER_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        // https://github.com/gfx-rs/wgpu/issues/7854
        TestParameters::default().skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_sync(|ctx| {
        let buffer_source = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let buffer_dest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);

        buffer_source.destroy();
        buffer_dest.destroy();

        let cmd_buffer = encoder.finish();

        fail(
            &ctx.device,
            || ctx.queue.submit([cmd_buffer]),
            Some("Buffer with '' label has been destroyed"),
        );
    });

// Test that destroying a texture between command buffer recording and
// submission fails gracefully.
#[gpu_test]
static TEXTURE_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        // https://github.com/gfx-rs/wgpu/issues/7854
        TestParameters::default().skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_sync(|ctx| {
        let descriptor = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // multisampling is not supported for clear
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let texture_1 = ctx.device.create_texture(&descriptor);
        let texture_2 = ctx.device.create_texture(&descriptor);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_1,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture_2,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );

        texture_1.destroy();
        texture_2.destroy();

        let cmd_buffer = encoder.finish();

        fail(
            &ctx.device,
            || ctx.queue.submit([cmd_buffer]),
            Some("Texture with '' label has been destroyed"),
        );
    });
