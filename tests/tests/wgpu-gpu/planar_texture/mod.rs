//! Tests for nv12 texture creation and sampling.

use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        NV12_TEXTURE_CREATION_SAMPLING,
        P010_TEXTURE_CREATION_SAMPLING,
        NV12_TEXTURE_RENDERING,
        NV12_TEXTURE_COPYING,
        P010_TEXTURE_COPYING,
        NV12_PLANE_TO_SINGLE_PLANE_COPY,
        NV12_ALL_PLANES_PARTIAL_COPY,
        P010_ALL_PLANES_PARTIAL_COPY,
        NV12_COPY_INTO_PLANE,
        P010_COPY_INTO_PLANE,
    ]);
}

// Helper function to test planar texture creation and sampling.
fn test_planar_texture_creation_sampling(
    ctx: &TestingContext,
    y_view: &wgpu::TextureView,
    uv_view: &wgpu::TextureView,
) {
    let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("planar_texture_sampling.wgsl"));
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planar texture pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        min_filter: wgpu::FilterMode::Linear,
        mag_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(y_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(uv_view),
            },
        ],
    });

    let target_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: y_view.texture().size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            ops: wgpu::Operations::default(),
            resolve_target: None,
            view: &target_view,
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..4, 0..1);
    drop(rpass);
    ctx.queue.submit([encoder.finish()]);
}

// Helper function to test rendering onto planar texture.
fn test_planar_texture_rendering(
    ctx: &TestingContext,
    (y_view, y_format): (&wgpu::TextureView, wgpu::TextureFormat),
    (uv_view, uv_format): (&wgpu::TextureView, wgpu::TextureFormat),
) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("planar_texture_rendering.wgsl"));
    let y_pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("y plane pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_y_main"),
                compilation_options: Default::default(),
                targets: &[Some(y_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let uv_pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uv plane pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_uv_main"),
                compilation_options: Default::default(),
                targets: &[Some(uv_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: y_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&y_pipeline);
        rpass.draw(0..3, 0..1);
    }
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: uv_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&uv_pipeline);
        rpass.draw(0..3, 0..1);
    }

    ctx.queue.submit([encoder.finish()]);
}

/// Ensures that creation and sampling of an NV12 format texture works as
/// expected.
#[apply(gpu_test!)]
static NV12_TEXTURE_CREATION_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R8Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rg8Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_creation_sampling(&ctx, &y_view, &uv_view);
    });

/// Ensures that creation and sampling of a P010 format texture works as
/// expected.
#[apply(gpu_test!)]
static P010_TEXTURE_CREATION_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            )
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::P010,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R16Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rg16Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_creation_sampling(&ctx, &y_view, &uv_view);
    });

/// Ensures that rendering on to NV12 format texture works as expected.
#[apply(gpu_test!)]
static NV12_TEXTURE_RENDERING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R8Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rg8Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_rendering(
            &ctx,
            (&y_view, wgpu::TextureFormat::R8Unorm),
            (&uv_view, wgpu::TextureFormat::Rg8Unorm),
        );
    });

/// Ensures that copying NV12 texture to NV12 texture works as expected
#[apply(gpu_test!)]
static NV12_TEXTURE_COPYING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let input_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let mut command_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        command_encoder.copy_texture_to_texture(
            input_texture.as_image_copy(),
            output_texture.as_image_copy(),
            size,
        );
        ctx.queue.submit([command_encoder.finish()]);
    });

/// Ensures that copying a single plane of an NV12 source into a matching
/// single-plane destination (Plane0 → R8Unorm, Plane1 → Rg8Unorm) round-trips
/// byte-for-byte. Exercises the planar→single-plane copy-compatibility
/// extension in `copy_texture_to_texture`.
#[apply(gpu_test!)]
static NV12_PLANE_TO_SINGLE_PLANE_COPY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_async(|ctx| async move {
        // Width chosen so that bytes-per-row is 256-aligned for both planes:
        //   luma   R8Unorm:  256 px * 1 byte/px = 256
        //   chroma Rg8Unorm: 128 px * 2 byte/px = 256
        // Height is deliberately different from width so a swap is caught.
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 128;
        let luma_size = wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        };
        let chroma_size = wgpu::Extent3d {
            width: WIDTH / 2,
            height: HEIGHT / 2,
            depth_or_array_layers: 1,
        };

        let nv12 = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nv12 src"),
            dimension: wgpu::TextureDimension::D2,
            size: luma_size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        // Distinct patterns per plane so a swap or plane-0-fallback would fail
        // the assertion at the end. Each pattern depends on both `x` and `y`
        // with different per-axis weights, so it is asymmetric under transpose,
        // horizontal mirror, and vertical mirror — a row/column mix-up changes
        // the bytes and fails the comparison.
        let luma_bytes: Vec<u8> = (0..HEIGHT)
            .flat_map(|y| {
                (0..WIDTH).map(move |x| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(101)) as u8)
            })
            .collect();
        let chroma_bytes: Vec<u8> = (0..HEIGHT / 2)
            .flat_map(|y| {
                (0..WIDTH / 2).flat_map(move |x| {
                    let r = x.wrapping_mul(7).wrapping_add(y.wrapping_mul(53)) as u8;
                    let g = (x.wrapping_mul(29).wrapping_add(y.wrapping_mul(13)) ^ 0xA5) as u8;
                    [r, g]
                })
            })
            .collect();

        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::Plane0,
            },
            &luma_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH),
                rows_per_image: Some(HEIGHT),
            },
            luma_size,
        );
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::Plane1,
            },
            &chroma_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH / 2 * 2),
                rows_per_image: Some(HEIGHT / 2),
            },
            chroma_size,
        );

        let r8 = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("r8 dst"),
            dimension: wgpu::TextureDimension::D2,
            size: luma_size,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let rg8 = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rg8 dst"),
            dimension: wgpu::TextureDimension::D2,
            size: chroma_size,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let r8_readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: luma_bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rg8_readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: chroma_bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // The path under test.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::Plane0,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &r8,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            luma_size,
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::Plane1,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &rg8,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            chroma_size,
        );

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &r8,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &r8_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH),
                    rows_per_image: Some(HEIGHT),
                },
            },
            luma_size,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &rg8,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rg8_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH / 2 * 2),
                    rows_per_image: Some(HEIGHT / 2),
                },
            },
            chroma_size,
        );

        ctx.queue.submit([encoder.finish()]);

        let r8_slice = r8_readback.slice(..);
        r8_slice.map_async(wgpu::MapMode::Read, |_| ());
        let rg8_slice = rg8_readback.slice(..);
        rg8_slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        let r8_data: Vec<u8> = r8_slice.get_mapped_range().unwrap().to_vec();
        let rg8_data: Vec<u8> = rg8_slice.get_mapped_range().unwrap().to_vec();
        assert_eq!(r8_data, luma_bytes, "luma plane mismatch");
        assert_eq!(rg8_data, chroma_bytes, "chroma plane mismatch");
    });

const PLANAR_COPY_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 256,
    height: 128,
    depth_or_array_layers: 1,
};

/// Size of a plane of a `PLANAR_COPY_SIZE` texture.
struct PlaneLayout {
    width: u32,
    height: u32,
    texel_size: u32,
}

impl PlaneLayout {
    fn new(format: wgpu::TextureFormat, plane: u32) -> Self {
        let aspect = wgpu::TextureAspect::from_plane(plane).unwrap();
        let plane_format = format.aspect_specific_format(aspect).unwrap();
        let (w_subsampling, h_subsampling) = format.subsampling_factors(Some(plane));
        Self {
            width: PLANAR_COPY_SIZE.width / w_subsampling,
            height: PLANAR_COPY_SIZE.height / h_subsampling,
            texel_size: plane_format.block_copy_size(None).unwrap(),
        }
    }

    fn row_bytes(&self) -> u32 {
        self.width * self.texel_size
    }

    fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }

    /// Copies a rectangle of texels from `src` into `dst`, both laid out as this plane.
    fn copy_rect(
        &self,
        src: &[u8],
        (src_x, src_y): (u32, u32),
        dst: &mut [u8],
        (dst_x, dst_y): (u32, u32),
        (width, height): (u32, u32),
    ) {
        let row_bytes = self.row_bytes() as usize;
        let ts = self.texel_size as usize;
        let len = width as usize * ts;
        for row in 0..height as usize {
            let src_start = (src_y as usize + row) * row_bytes + src_x as usize * ts;
            let dst_start = (dst_y as usize + row) * row_bytes + dst_x as usize * ts;
            dst[dst_start..dst_start + len].copy_from_slice(&src[src_start..src_start + len]);
        }
    }
}

fn planar_copy_texture(
    ctx: &TestingContext,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        dimension: wgpu::TextureDimension::D2,
        size,
        format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    })
}

/// Distinct, position-dependent bytes for each `seed`, so that misplaced or missing
/// texels change the result.
fn planar_copy_pattern(seed: u32, len: u32) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_add(seed << 20).wrapping_mul(2654435761) >> 24) as u8)
        .collect()
}

/// Fills `aspect` of `texture` with a pattern laid out as `plane`, and returns the pattern.
fn planar_copy_fill(
    ctx: &TestingContext,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    plane: &PlaneLayout,
    seed: u32,
) -> Vec<u8> {
    let data = planar_copy_pattern(seed, plane.row_bytes() * plane.height);
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(plane.row_bytes()),
            rows_per_image: None,
        },
        plane.extent(),
    );
    data
}

async fn planar_copy_read(
    ctx: &TestingContext,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    plane: &PlaneLayout,
) -> Vec<u8> {
    let padded_row_bytes = plane
        .row_bytes()
        .next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded_row_bytes * plane.height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: None,
            },
        },
        plane.extent(),
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    mapped
        .chunks(padded_row_bytes as usize)
        .flat_map(|row| &row[..plane.row_bytes() as usize])
        .copied()
        .collect()
}

fn texel_copy_info(
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    (x, y): (u32, u32),
) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x, y, z: 0 },
        aspect,
    }
}

/// Copies a partial, offset region of all planes between two multi-planar textures and
/// checks that each plane received the correspondingly subsampled region.
async fn planar_all_planes_partial_copy(ctx: TestingContext, format: wgpu::TextureFormat) {
    use wgpu::TextureAspect as Ta;

    let planes = [PlaneLayout::new(format, 0), PlaneLayout::new(format, 1)];
    let aspects = [Ta::Plane0, Ta::Plane1];

    let src = planar_copy_texture(&ctx, format, PLANAR_COPY_SIZE);
    let dst = planar_copy_texture(&ctx, format, PLANAR_COPY_SIZE);
    let mut expected = Vec::new();
    for (i, (plane, aspect)) in planes.iter().zip(aspects).enumerate() {
        let src_data = planar_copy_fill(&ctx, &src, aspect, plane, 1 + i as u32);
        let dst_data = planar_copy_fill(&ctx, &dst, aspect, plane, 3 + i as u32);
        expected.push((src_data, dst_data));
    }

    let (src_origin, dst_origin, size) = ((4, 2), (2, 6), (240, 112));
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_texture(
        texel_copy_info(&src, Ta::All, src_origin),
        texel_copy_info(&dst, Ta::All, dst_origin),
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    for (i, (plane, aspect)) in planes.iter().zip(aspects).enumerate() {
        let (w_sub, h_sub) = format.subsampling_factors(Some(i as u32));
        let sub = |(x, y): (u32, u32)| (x / w_sub, y / h_sub);
        let (src_data, dst_data) = &mut expected[i];
        plane.copy_rect(
            src_data,
            sub(src_origin),
            dst_data,
            sub(dst_origin),
            sub(size),
        );

        let actual = planar_copy_read(&ctx, &dst, aspect, plane).await;
        assert!(actual == *dst_data, "plane {i} mismatch");
    }
}

/// Ensures that copying a partial, offset region of all planes of an NV12 texture
/// copies the subsampled region of each plane.
#[apply(gpu_test!)]
static NV12_ALL_PLANES_PARTIAL_COPY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_async(|ctx| planar_all_planes_partial_copy(ctx, wgpu::TextureFormat::NV12));

/// Ensures that copying a partial, offset region of all planes of a P010 texture
/// copies the subsampled region of each plane.
#[apply(gpu_test!)]
static P010_ALL_PLANES_PARTIAL_COPY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(
            wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        ))
        .run_async(|ctx| planar_all_planes_partial_copy(ctx, wgpu::TextureFormat::P010));

/// Copies regions into individual planes of a multi-planar texture, from single-plane
/// textures and from the planes of another multi-planar texture, and checks that only
/// the targeted region of the targeted plane changed.
async fn planar_copy_into_plane(ctx: TestingContext, format: wgpu::TextureFormat) {
    use wgpu::TextureAspect as Ta;

    let planes = [PlaneLayout::new(format, 0), PlaneLayout::new(format, 1)];
    let aspects = [Ta::Plane0, Ta::Plane1];

    let dst = planar_copy_texture(&ctx, format, PLANAR_COPY_SIZE);
    let planar_src = planar_copy_texture(&ctx, format, PLANAR_COPY_SIZE);
    let mut dst_data = Vec::new();
    let mut planar_src_data = Vec::new();
    let mut single_srcs = Vec::new();
    for (i, (plane, aspect)) in planes.iter().zip(aspects).enumerate() {
        dst_data.push(planar_copy_fill(&ctx, &dst, aspect, plane, 1 + i as u32));
        planar_src_data.push(planar_copy_fill(
            &ctx,
            &planar_src,
            aspect,
            plane,
            3 + i as u32,
        ));

        let single_format = format.aspect_specific_format(aspect).unwrap();
        let single = planar_copy_texture(&ctx, single_format, plane.extent());
        let single_data = planar_copy_fill(&ctx, &single, Ta::All, plane, 5 + i as u32);
        single_srcs.push((single, single_data));
    }

    // (plane, copy from the multi-planar source, source origin, destination origin, size)
    let copies = [
        (0, false, (8, 4), (16, 2), (64, 32)),
        (1, false, (4, 2), (8, 10), (32, 16)),
        (0, true, (0, 2), (100, 80), (40, 20)),
        (1, true, (10, 0), (60, 40), (20, 12)),
    ];

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (plane, from_planar, src_origin, dst_origin, size) in copies {
        let source = if from_planar {
            texel_copy_info(&planar_src, aspects[plane], src_origin)
        } else {
            texel_copy_info(&single_srcs[plane].0, Ta::All, src_origin)
        };
        encoder.copy_texture_to_texture(
            source,
            texel_copy_info(&dst, aspects[plane], dst_origin),
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        let src_data = if from_planar {
            &planar_src_data[plane]
        } else {
            &single_srcs[plane].1
        };
        planes[plane].copy_rect(src_data, src_origin, &mut dst_data[plane], dst_origin, size);
    }
    ctx.queue.submit([encoder.finish()]);

    for (i, (plane, aspect)) in planes.iter().zip(aspects).enumerate() {
        let actual = planar_copy_read(&ctx, &dst, aspect, plane).await;
        assert!(actual == dst_data[i], "plane {i} mismatch");
    }
}

/// Ensures that copies into a single plane of an NV12 texture only modify that plane.
#[apply(gpu_test!)]
static NV12_COPY_INTO_PLANE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_async(|ctx| planar_copy_into_plane(ctx, wgpu::TextureFormat::NV12));

/// Ensures that copies into a single plane of a P010 texture only modify that plane.
#[apply(gpu_test!)]
static P010_COPY_INTO_PLANE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(
            wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        ))
        .run_async(|ctx| planar_copy_into_plane(ctx, wgpu::TextureFormat::P010));

/// Ensures that copying P010 texture to P010 texture works as expected
#[apply(gpu_test!)]
static P010_TEXTURE_COPYING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            )
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = wgpu::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let input_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::P010,
            usage: wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::P010,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let mut command_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        command_encoder.copy_texture_to_texture(
            input_texture.as_image_copy(),
            output_texture.as_image_copy(),
            size,
        );
        ctx.queue.submit([command_encoder.finish()]);
    });
