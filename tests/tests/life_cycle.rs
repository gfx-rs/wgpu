use wgpu_test::{fail, gpu_test, GpuTestConfiguration};

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

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

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

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

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
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        {
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            ctx.async_poll(wgpu::Maintain::wait())
                .await
                .panic_on_timeout();
            buffer.destroy();
        }
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
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

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

        texture.destroy();

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

        texture.destroy();

        texture.destroy();
    });
