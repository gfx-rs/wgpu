use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(CHILD_TEXTURE_PRESERVES_INIT_STATE_AND_CONTENT_AFTER_DESTROY);
}

fn texture_descriptor(
    usage: wgpu::TextureUsages,
) -> wgpu::wgc::resource::TextureDescriptor<'static> {
    wgpu::wgc::resource::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: Vec::new(),
    }
}

#[apply(gpu_test!)]
static CHILD_TEXTURE_PRESERVES_INIT_STATE_AND_CONTENT_AFTER_DESTROY: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let parent = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let core_parent = parent.as_core().unwrap();
        let child = wgpu::Texture::from_core(
            core_parent
                .create_child_texture(&texture_descriptor(wgpu::TextureUsages::COPY_DST))
                .unwrap(),
        );

        let data = [0x5a; 4 * 4 * 4];
        ctx.queue.write_texture(
            child.as_image_copy(),
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        child.destroy();
        drop(child);

        let child = wgpu::Texture::from_core(
            core_parent
                .create_child_texture(&texture_descriptor(wgpu::TextureUsages::COPY_SRC))
                .unwrap(),
        );
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            child.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(4),
                },
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, Result::unwrap);
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        for row in 0..4 {
            assert_eq!(&mapped[row * 256..row * 256 + 16], &[0x5a; 16]);
        }
        drop(mapped);
        readback.unmap();
        child.destroy();
    });
