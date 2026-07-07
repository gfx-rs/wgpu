//! Tests for zero-initialization of resources.
//!
//! It is common for allocations on a fresh heap to coincidentally be zero, which can cause
//! these tests to produce false negatives. One way to make them more reliable is to run
//! them on llvmpipe with `LVP_POISON_MEMORY=true` in the environment.

use core::num::NonZeroU64;

use wgpu::*;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

/// A way to write data into a texture.
#[derive(Clone, Copy)]
#[allow(clippy::enum_variant_names)]
enum WriteMethod {
    WriteTexture,
    CopyBufferToTexture,
    CopyTextureToTexture,
}

impl WriteMethod {
    fn name(self) -> &'static str {
        match self {
            WriteMethod::WriteTexture => "write_texture",
            WriteMethod::CopyBufferToTexture => "copy_buffer_to_texture",
            WriteMethod::CopyTextureToTexture => "copy_texture_to_texture",
        }
    }
}

/// A way to read data out of a texture.
#[derive(Clone, Copy)]
enum ReadMethod {
    CopyTextureToBuffer,
    CopyTextureToTexture,
}

impl ReadMethod {
    fn name(self) -> &'static str {
        match self {
            ReadMethod::CopyTextureToBuffer => "copy_texture_to_buffer",
            ReadMethod::CopyTextureToTexture => "copy_texture_to_texture",
        }
    }
}

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COPY_BUFFER_TO_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_NV12,
        COPY_BUFFER_TO_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH32FLOAT_STENCIL8,
        DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_AFTER_SUBMIT,
        DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_IN_SAME_ENCODER,
        DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_IN_SAME_ENCODER,
        DISCARDING_EITHER_DEPTH_OR_STENCIL_ASPECT_TEST,
        WRITE_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_NV12,
        WRITE_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_P010,
        WRITE_TEXTURE_PLANE1_LEAVES_PLANE0_UNINIT_NV12,
        WRITE_TEXTURE_PLANE1_LEAVES_PLANE0_UNINIT_P010,
        WRITE_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH24PLUS_STENCIL8,
        WRITE_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH32FLOAT_STENCIL8,
        DYNAMIC_OFFSET_BUFFER_BINDING_INIT,
        COPY_TEXTURE_TO_BUFFER_3D_SOURCE_ORIGIN_Z_UNINIT,
        COPY_TEXTURE_TO_TEXTURE_3D_SOURCE_ORIGIN_Z_UNINIT,
        COPY_BUFFER_TO_TEXTURE_3D_DEST_ORIGIN_Z_PARTIAL,
        COPY_TEXTURE_TO_TEXTURE_3D_DEST_ORIGIN_Z_PARTIAL,
    ]);
}

// Checks if discarding a color target resets its init state, causing a zero read of this texture when copied in after submit of the encoder.
#[gpu_test]
static DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_AFTER_SUBMIT:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
    .run_async(|mut ctx| async move {
        let mut case = DiscardTestCase::new(&mut ctx, TextureFormat::Rgba8UnormSrgb);
        case.create_command_encoder();
        case.discard();
        case.submit_command_encoder();

        case.create_command_encoder();
        case.copy_texture_to_buffer();
        case.submit_command_encoder();

        case.assert_buffers_are_zero().await;
    });

#[gpu_test]
static DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_IN_SAME_ENCODER:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
    .run_async(|mut ctx| async move {
        let mut case = DiscardTestCase::new(&mut ctx, TextureFormat::Rgba8UnormSrgb);
        case.create_command_encoder();
        case.discard();
        case.copy_texture_to_buffer();
        case.submit_command_encoder();

        case.assert_buffers_are_zero().await;
    });

#[gpu_test]
static DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE_CHECK_VISIBLE_ON_COPY_IN_SAME_ENCODER:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(
                DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES | DownlevelFlags::COMPUTE_SHADERS,
            )
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|mut ctx| async move {
        for format in [
            TextureFormat::Stencil8,
            TextureFormat::Depth16Unorm,
            TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8,
            TextureFormat::Depth32Float,
        ] {
            let mut case = DiscardTestCase::new(&mut ctx, format);
            case.create_command_encoder();
            case.discard();
            case.copy_texture_to_buffer();
            case.submit_command_encoder();

            case.assert_buffers_are_zero().await;
        }
    });

#[gpu_test]
static DISCARDING_EITHER_DEPTH_OR_STENCIL_ASPECT_TEST: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(
                    DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES
                        | DownlevelFlags::COMPUTE_SHADERS,
                )
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|mut ctx| async move {
            for format in [
                TextureFormat::Stencil8,
                TextureFormat::Depth16Unorm,
                TextureFormat::Depth24Plus,
                TextureFormat::Depth24PlusStencil8,
                TextureFormat::Depth32Float,
            ] {
                let mut case = DiscardTestCase::new(&mut ctx, format);
                case.create_command_encoder();
                case.discard_depth();
                case.submit_command_encoder();

                case.create_command_encoder();
                case.discard_stencil();
                case.submit_command_encoder();

                case.create_command_encoder();
                case.copy_texture_to_buffer();
                case.submit_command_encoder();

                case.assert_buffers_are_zero().await;
            }
        });

struct DiscardTestCase<'ctx> {
    ctx: &'ctx mut TestingContext,
    format: TextureFormat,
    texture: Texture,
    readback_buffers: ReadbackBuffers,
    encoder: Option<CommandEncoder>,
}

impl<'ctx> DiscardTestCase<'ctx> {
    pub fn new(ctx: &'ctx mut TestingContext, format: TextureFormat) -> Self {
        let extra_usages = match format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                TextureUsages::TEXTURE_BINDING
            }
            _ => TextureUsages::empty(),
        };

        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("RenderTarget"),
            // Non-square so a width/height swap in the row-stride math is caught.
            // `width` is a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` so
            // `bytes_per_row` stays aligned for every tested format.
            size: Extent3d {
                width: COPY_BYTES_PER_ROW_ALIGNMENT,
                height: COPY_BYTES_PER_ROW_ALIGNMENT / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | extra_usages,
            view_formats: &[],
        });

        // Clear using a write_texture operation. We could also clear using a render_pass clear.
        // However, when making this test intentionally fail (by breaking wgpu impl), it shows that at least on the tested Vulkan driver,
        // the later following discard pass in the test (i.e. internally vk::AttachmentStoreOp::DONT_CARE) will yield different depending on the operation we take here:
        // * clearing white -> discard will cause it to become black!
        // * clearing red -> discard will keep it red
        // * write_texture -> discard will keep buffer
        // This behavior is curious, but does not violate any spec - it is wgpu's job to pass this test no matter what a render target discard does.

        // ... but that said, for depth/stencil textures we need to do a clear.
        if format.is_depth_stencil_format() {
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Depth/Stencil setup"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &texture.create_view(&TextureViewDescriptor::default()),
                    depth_ops: format.has_depth_aspect().then_some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: format.has_stencil_aspect().then_some(Operations {
                        load: LoadOp::Clear(0xFFFFFFFF),
                        store: StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            ctx.queue.submit([encoder.finish()]);
        } else {
            let block_size = format.block_copy_size(None).unwrap();
            let bytes_per_row = texture.width() * block_size;

            // Size for tests is chosen so that we don't need to care about buffer alignments.
            assert!(!format.is_compressed());
            assert_eq!(bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);

            let buffer_size = texture.height() * bytes_per_row;
            let data = vec![255; buffer_size as usize];
            ctx.queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                &data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
                texture.size(),
            );
        }

        let readback_buffers = ReadbackBuffers::new(&ctx.device, &texture);

        Self {
            ctx,
            format,
            texture,
            readback_buffers,
            encoder: None,
        }
    }

    pub fn create_command_encoder(&mut self) {
        self.encoder = Some(
            self.ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default()),
        )
    }

    pub fn submit_command_encoder(&mut self) {
        self.ctx
            .queue
            .submit([self.encoder.take().unwrap().finish()]);
    }

    pub fn discard(&mut self) {
        self.encoder
            .as_mut()
            .unwrap()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("Discard"),
                color_attachments: &[self.format.has_color_aspect().then_some(
                    RenderPassColorAttachment {
                        view: &self.texture.create_view(&TextureViewDescriptor::default()),
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Discard,
                        },
                    },
                )],
                depth_stencil_attachment: self.format.is_depth_stencil_format().then_some(
                    RenderPassDepthStencilAttachment {
                        view: &self.texture.create_view(&TextureViewDescriptor::default()),
                        depth_ops: self.format.has_depth_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Discard,
                        }),
                        stencil_ops: self.format.has_stencil_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Discard,
                        }),
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
    }

    pub fn discard_depth(&mut self) {
        self.encoder
            .as_mut()
            .unwrap()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("Discard Depth"),
                color_attachments: &[],
                depth_stencil_attachment: self.format.is_depth_stencil_format().then_some(
                    RenderPassDepthStencilAttachment {
                        view: &self.texture.create_view(&TextureViewDescriptor::default()),
                        depth_ops: self.format.has_depth_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Discard,
                        }),
                        stencil_ops: self.format.has_stencil_aspect().then_some(Operations {
                            load: LoadOp::Clear(0),
                            store: StoreOp::Store,
                        }),
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
    }

    pub fn discard_stencil(&mut self) {
        self.encoder
            .as_mut()
            .unwrap()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("Discard Stencil"),
                color_attachments: &[],
                depth_stencil_attachment: self.format.is_depth_stencil_format().then_some(
                    RenderPassDepthStencilAttachment {
                        view: &self.texture.create_view(&TextureViewDescriptor::default()),
                        depth_ops: self.format.has_depth_aspect().then_some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: self.format.has_stencil_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Discard,
                        }),
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
    }

    pub fn copy_texture_to_buffer(&mut self) {
        self.readback_buffers.copy_from(
            &self.ctx.device,
            self.encoder.as_mut().unwrap(),
            &self.texture,
        );
    }

    pub async fn assert_buffers_are_zero(&mut self) {
        assert!(
            self.readback_buffers.are_zero(self.ctx).await,
            "texture was not fully cleared"
        );
    }
}

// Tests that a full-extent, single-aspect `write_texture` does not cause
// *other* aspects of a multi-aspect texture to be considered initialized.

#[gpu_test]
static WRITE_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH32FLOAT_STENCIL8: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::DEPTH32FLOAT_STENCIL8)
                .downlevel_flags(DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            check_depth_stencil_write_leaves_other_uninit(
                &ctx,
                TextureFormat::Depth32FloatStencil8,
                /* depth_bpp */ 4,
                TextureAspect::StencilOnly,
                WriteMethod::WriteTexture,
            )
            .await;
        });

// Note: there aren't corresponding `WRITE_TEXTURE_DEPTH_LEAVES_STENCIL_UNINIT_*`
// cases because the depth aspect of the combined depth/stencil formats cannot
// be the destination of a `write_texture` call.
#[gpu_test]
static WRITE_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH24PLUS_STENCIL8: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(
                    DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES
                        | DownlevelFlags::COMPUTE_SHADERS,
                )
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            // Non-square so a width/height swap is caught.
            let size = Extent3d {
                width: 256,
                height: 192,
                depth_or_array_layers: 1,
            };

            // Depth aspect of Depth24PlusStencil8 cannot be the source of a direct
            // copy_texture_to_buffer, so we cannot use the same readback strategy
            // as the other depth/stencil format. Use the shared ReadbackBuffers
            // helper, which reads the depth aspect through a compute shader.
            // Because that helper checks both aspects, we write zeros (not a
            // sentinel byte) to the stencil aspect.
            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: Some("depth24plus-stencil8 aspect-init test"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth24PlusStencil8,
                usage: TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC
                    | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let stencil_bytes_per_row = size.width;
            let stencil_data = vec![0u8; (stencil_bytes_per_row * size.height) as usize];
            ctx.queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::StencilOnly,
                },
                &stencil_data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stencil_bytes_per_row),
                    rows_per_image: Some(size.height),
                },
                size,
            );
            ctx.queue.submit(None);

            let readback_buffers = ReadbackBuffers::new(&ctx.device, &texture);
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            readback_buffers.copy_from(&ctx.device, &mut encoder, &texture);
            ctx.queue.submit([encoder.finish()]);

            assert!(
                readback_buffers.are_zero(&ctx).await,
                "depth aspect of Depth24PlusStencil8 read back non-zero after \
             stencil-only write_texture",
            );
        });

#[gpu_test]
static WRITE_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_NV12: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::TEXTURE_FORMAT_NV12)
                .limits(Limits::downlevel_defaults())
                // https://github.com/gfx-rs/wgpu/issues/9688
                .expect_fail(FailureCase::lvp_poison_memory("read back non-zero")),
        )
        .run_async(|ctx| async move {
            check_plane_write_leaves_other_plane_uninit(
                &ctx,
                TextureFormat::NV12,
                /* plane0_bpp */ 1,
                /* plane1_bpp */ 2,
                TextureAspect::Plane0,
                WriteMethod::WriteTexture,
            )
            .await;
        });

#[gpu_test]
static WRITE_TEXTURE_PLANE1_LEAVES_PLANE0_UNINIT_NV12: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::TEXTURE_FORMAT_NV12)
                .limits(Limits::downlevel_defaults())
                // https://github.com/gfx-rs/wgpu/issues/9688
                .expect_fail(FailureCase::lvp_poison_memory("read back non-zero")),
        )
        .run_async(|ctx| async move {
            check_plane_write_leaves_other_plane_uninit(
                &ctx,
                TextureFormat::NV12,
                /* plane0_bpp */ 1,
                /* plane1_bpp */ 2,
                TextureAspect::Plane1,
                WriteMethod::WriteTexture,
            )
            .await;
        });

#[gpu_test]
static WRITE_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_P010: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::TEXTURE_FORMAT_P010 | Features::TEXTURE_FORMAT_16BIT_NORM)
                .limits(Limits::downlevel_defaults())
                // https://github.com/gfx-rs/wgpu/issues/9688
                .expect_fail(FailureCase::lvp_poison_memory("read back non-zero")),
        )
        .run_async(|ctx| async move {
            check_plane_write_leaves_other_plane_uninit(
                &ctx,
                TextureFormat::P010,
                /* plane0_bpp */ 2,
                /* plane1_bpp */ 4,
                TextureAspect::Plane0,
                WriteMethod::WriteTexture,
            )
            .await;
        });

#[gpu_test]
static WRITE_TEXTURE_PLANE1_LEAVES_PLANE0_UNINIT_P010: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::TEXTURE_FORMAT_P010 | Features::TEXTURE_FORMAT_16BIT_NORM)
                .limits(Limits::downlevel_defaults())
                // https://github.com/gfx-rs/wgpu/issues/9688
                .expect_fail(FailureCase::lvp_poison_memory("read back non-zero")),
        )
        .run_async(|ctx| async move {
            check_plane_write_leaves_other_plane_uninit(
                &ctx,
                TextureFormat::P010,
                /* plane0_bpp */ 2,
                /* plane1_bpp */ 4,
                TextureAspect::Plane1,
                WriteMethod::WriteTexture,
            )
            .await;
        });

// The write_texture tests exhaustively cover all the relevant format/aspect combinations.
// These copy_buffer_to_texture tests only sanity-check one depth/stencil format and one
// multi-planar format.
#[gpu_test]
static COPY_BUFFER_TO_TEXTURE_STENCIL_LEAVES_DEPTH_UNINIT_DEPTH32FLOAT_STENCIL8:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::DEPTH32FLOAT_STENCIL8)
            .downlevel_flags(DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        check_depth_stencil_write_leaves_other_uninit(
            &ctx,
            TextureFormat::Depth32FloatStencil8,
            /* depth_bpp */ 4,
            TextureAspect::StencilOnly,
            WriteMethod::CopyBufferToTexture,
        )
        .await;
    });

#[gpu_test]
static COPY_BUFFER_TO_TEXTURE_PLANE0_LEAVES_PLANE1_UNINIT_NV12: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::TEXTURE_FORMAT_NV12)
                .limits(Limits::downlevel_defaults())
                // https://github.com/gfx-rs/wgpu/issues/9688
                .expect_fail(FailureCase::lvp_poison_memory("read back non-zero")),
        )
        .run_async(|ctx| async move {
            check_plane_write_leaves_other_plane_uninit(
                &ctx,
                TextureFormat::NV12,
                /* plane0_bpp */ 1,
                /* plane1_bpp */ 2,
                TextureAspect::Plane0,
                WriteMethod::CopyBufferToTexture,
            )
            .await;
        });

struct AspectInfo {
    aspect: TextureAspect,
    size: Extent3d,
    bpp: u32,
}

async fn check_depth_stencil_write_leaves_other_uninit(
    ctx: &TestingContext,
    format: TextureFormat,
    depth_bpp: u32,
    write_aspect: TextureAspect,
    method: WriteMethod,
) {
    // Non-square so a width/height swap is caught.
    let size = Extent3d {
        width: 256,
        height: 192,
        depth_or_array_layers: 1,
    };
    let (write_bpp, read_aspect, read_bpp) = match write_aspect {
        TextureAspect::StencilOnly => (1, TextureAspect::DepthOnly, depth_bpp),
        TextureAspect::DepthOnly => (depth_bpp, TextureAspect::StencilOnly, 1),
        _ => panic!("expected DepthOnly or StencilOnly"),
    };
    check_write_aspect_leaves_other_uninit(
        ctx,
        format,
        AspectInfo {
            aspect: write_aspect,
            size,
            bpp: write_bpp,
        },
        AspectInfo {
            aspect: read_aspect,
            size,
            bpp: read_bpp,
        },
        method,
    )
    .await;
}

async fn check_plane_write_leaves_other_plane_uninit(
    ctx: &TestingContext,
    format: TextureFormat,
    plane0_bpp: u32,
    plane1_bpp: u32,
    write_plane: TextureAspect,
    method: WriteMethod,
) {
    // Plane 1 of NV12/P010 is half resolution in each dimension.
    // Non-square so a width/height swap is caught.
    let full_size = Extent3d {
        width: 256,
        height: 192,
        depth_or_array_layers: 1,
    };
    let half_size = Extent3d {
        width: 128,
        height: 96,
        depth_or_array_layers: 1,
    };
    let (write_size, write_bpp, read_aspect, read_size, read_bpp) = match write_plane {
        TextureAspect::Plane0 => (
            full_size,
            plane0_bpp,
            TextureAspect::Plane1,
            half_size,
            plane1_bpp,
        ),
        TextureAspect::Plane1 => (
            half_size,
            plane1_bpp,
            TextureAspect::Plane0,
            full_size,
            plane0_bpp,
        ),
        _ => panic!("expected Plane0 or Plane1"),
    };
    check_write_aspect_leaves_other_uninit(
        ctx,
        format,
        AspectInfo {
            aspect: write_plane,
            size: write_size,
            bpp: write_bpp,
        },
        AspectInfo {
            aspect: read_aspect,
            size: read_size,
            bpp: read_bpp,
        },
        method,
    )
    .await;
}

async fn check_write_aspect_leaves_other_uninit(
    ctx: &TestingContext,
    format: TextureFormat,
    write: AspectInfo,
    read: AspectInfo,
    method: WriteMethod,
) {
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("aspect-init test"),
        // Must match the full-plane / full-aspect size used by the callers.
        size: Extent3d {
            width: 256,
            height: 192,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let write_bytes_per_row = write.size.width * write.bpp;
    assert_eq!(write_bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let write_data = vec![0xAA_u8; (write_bytes_per_row * write.size.height) as usize];
    let write_layout = TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(write_bytes_per_row),
        rows_per_image: Some(write.size.height),
    };
    let write_texture_info = TexelCopyTextureInfo {
        texture: &texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: write.aspect,
    };
    match method {
        WriteMethod::WriteTexture => {
            ctx.queue
                .write_texture(write_texture_info, &write_data, write_layout, write.size);
            ctx.queue.submit(None);
        }
        WriteMethod::CopyBufferToTexture => {
            let src_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: Some("aspect-init source"),
                size: write_data.len() as u64,
                usage: BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = src_buffer.slice(..).get_mapped_range_mut().unwrap();
                view.copy_from_slice(&write_data);
            }
            src_buffer.unmap();

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: &src_buffer,
                    layout: write_layout,
                },
                write_texture_info,
                write.size,
            );
            ctx.queue.submit(Some(encoder.finish()));
        }
        WriteMethod::CopyTextureToTexture => {
            unreachable!("aspect-init tests do not use copy_texture_to_texture")
        }
    }

    let read_bytes_per_row = read.size.width * read.bpp;
    assert_eq!(read_bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let read_size_bytes = (read_bytes_per_row * read.size.height) as u64;
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("aspect readback"),
        size: read_size_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = readback.slice(..).get_mapped_range_mut().unwrap();
        let len = view.len();
        view.copy_from_slice(&vec![0xCD_u8; len]);
    }
    readback.unmap();

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: read.aspect,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(read_bytes_per_row),
                rows_per_image: Some(read.size.height),
            },
        },
        read.size,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

    let nonzero = data.iter().position(|&b| b != 0);
    assert!(
        nonzero.is_none(),
        "{:?} aspect of {:?} read back non-zero after {:?} {}; \
         first non-zero byte at offset {} = 0x{:02x}",
        read.aspect,
        format,
        write.aspect,
        method.name(),
        nonzero.unwrap(),
        data[nonzero.unwrap()],
    );
}

// Test that buffer ranges are properly initialized when used with a dynamic offset binding.
#[gpu_test]
static DYNAMIC_OFFSET_BUFFER_BINDING_INIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        // `OFFSET` must be aligned to minStorageBufferOffsetAlignment; WebGPU guarantees 256.
        const OFFSET: u32 = 256;
        const BUFFER_SIZE: u64 = 4096;
        const BINDING_SIZE: u64 = 4;

        let input = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: BUFFER_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: BINDING_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: BINDING_SIZE,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader reads input[0] (which the dynamic offset shifts to `OFFSET / 4`) and writes it
        // to output[0].
        let shader_src = "
            @group(0) @binding(0) var<storage, read> input: array<u32, 1>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32, 1>;
            @compute @workgroup_size(1)
            fn main() {
                output[0] = input[0];
            }
            ";
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader_src.into()),
        });
        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &input,
                        offset: 0,
                        size: NonZeroU64::new(4),
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &output,
                        offset: 0,
                        size: NonZeroU64::new(4),
                    }),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[OFFSET]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, BINDING_SIZE);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

        let nonzero = data.iter().position(|&b| b != 0);
        assert!(
            nonzero.is_none(),
            "dynamic-offset bind group read back non-zero from unwritten \
                 region of a fresh storage buffer; first non-zero byte at offset \
                 {} = 0x{:02x}",
            nonzero.unwrap(),
            data[nonzero.unwrap()],
        );
    });

// Tests of initialization of 3D textures.
//
// Init tracking only operates on array layers, not on depth/volume slices
// of 3D textures. Therefore,

const D3_WIDTH: u32 = 256;
const D3_HEIGHT: u32 = 2;
const D3_DEPTH: u32 = 4;

// A read from a fresh 3D texture as a copy *source* at `origin.z >= 1` must
// trigger initialization of the full texture.
#[gpu_test]
static COPY_TEXTURE_TO_BUFFER_3D_SOURCE_ORIGIN_Z_UNINIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_source_init(&ctx, ReadMethod::CopyTextureToBuffer).await;
        });

#[gpu_test]
static COPY_TEXTURE_TO_TEXTURE_3D_SOURCE_ORIGIN_Z_UNINIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_source_init(&ctx, ReadMethod::CopyTextureToTexture).await;
        });

// The first depth slice must be initialized to zero before a partial copy into a fresh 3D
// texture with destination `origin.z >= 1`.
#[gpu_test]
static COPY_BUFFER_TO_TEXTURE_3D_DEST_ORIGIN_Z_PARTIAL: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_dest_init(&ctx, WriteMethod::CopyBufferToTexture).await;
        });

#[gpu_test]
static COPY_TEXTURE_TO_TEXTURE_3D_DEST_ORIGIN_Z_PARTIAL: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_dest_init(&ctx, WriteMethod::CopyTextureToTexture).await;
        });

fn create_3d_texture(ctx: &TestingContext, label: &str, depth: u32) -> Texture {
    ctx.device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: D3_WIDTH,
            height: D3_HEIGHT,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::R8Uint,
        usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn d3_buffer_layout() -> TexelCopyBufferLayout {
    TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(D3_WIDTH),
        rows_per_image: Some(D3_HEIGHT),
    }
}

async fn map_and_read(ctx: &TestingContext, buffer: &Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    slice.get_mapped_range().unwrap().to_vec()
}

async fn check_3d_copy_source_init(ctx: &TestingContext, method: ReadMethod) {
    const COPY_Z: u32 = 1;
    let copy_depth = D3_DEPTH - COPY_Z;
    let copy_size = Extent3d {
        width: D3_WIDTH,
        height: D3_HEIGHT,
        depth_or_array_layers: copy_depth,
    };

    let src = create_3d_texture(ctx, "3d source init test", D3_DEPTH);
    let src_info = TexelCopyTextureInfo {
        texture: &src,
        mip_level: 0,
        origin: Origin3d {
            x: 0,
            y: 0,
            z: COPY_Z,
        },
        aspect: TextureAspect::All,
    };

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    match method {
        ReadMethod::CopyTextureToBuffer => {
            // The copy source is partial (it starts at origin.z = COPY_Z), so the
            // readback can't go through `ReadbackBuffers`, which always copies the
            // full texture from the origin.
            let readback = ctx.device.create_buffer(&BufferDescriptor {
                label: Some("3d source readback"),
                size: (D3_WIDTH * D3_HEIGHT * copy_depth) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                src_info,
                TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: d3_buffer_layout(),
                },
                copy_size,
            );
            ctx.queue.submit(Some(encoder.finish()));

            let data = map_and_read(ctx, &readback).await;
            let nonzero = data.iter().position(|&b| b != 0);
            assert!(
                nonzero.is_none(),
                "3D texture used as {} source at origin.z={} read back non-zero from \
                 never-written memory; first non-zero byte at offset {} = 0x{:02x}",
                method.name(),
                COPY_Z,
                nonzero.unwrap(),
                data[nonzero.unwrap()],
            );
        }
        ReadMethod::CopyTextureToTexture => {
            let dst = create_3d_texture(ctx, "3d source init test dst", copy_depth);
            encoder.copy_texture_to_texture(
                src_info,
                TexelCopyTextureInfo {
                    texture: &dst,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                copy_size,
            );
            // The full `dst` is read back, so route it through `ReadbackBuffers`.
            let readback_buffers = ReadbackBuffers::new(&ctx.device, &dst);
            readback_buffers.copy_from(&ctx.device, &mut encoder, &dst);
            ctx.queue.submit(Some(encoder.finish()));

            assert!(
                readback_buffers.are_zero(ctx).await,
                "3D texture used as {} source at origin.z={} read back non-zero from \
                 never-written memory",
                method.name(),
                COPY_Z,
            );
        }
    }
}

async fn check_3d_copy_dest_init(ctx: &TestingContext, method: WriteMethod) {
    const DST_Z: u32 = 1;
    const SENTINEL: u8 = 0xAA;
    let slice_bytes = (D3_WIDTH * D3_HEIGHT) as usize;
    let one_slice = Extent3d {
        width: D3_WIDTH,
        height: D3_HEIGHT,
        depth_or_array_layers: 1,
    };

    let dst = create_3d_texture(ctx, "3d dest init test", D3_DEPTH);
    let dst_info = TexelCopyTextureInfo {
        texture: &dst,
        mip_level: 0,
        origin: Origin3d {
            x: 0,
            y: 0,
            z: DST_Z,
        },
        aspect: TextureAspect::All,
    };

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    match method {
        WriteMethod::CopyBufferToTexture => {
            let src_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: Some("3d dest init source"),
                size: slice_bytes as u64,
                usage: BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = src_buffer.slice(..).get_mapped_range_mut().unwrap();
                view.copy_from_slice(&vec![SENTINEL; slice_bytes]);
            }
            src_buffer.unmap();
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: &src_buffer,
                    layout: d3_buffer_layout(),
                },
                dst_info,
                one_slice,
            );
        }
        WriteMethod::CopyTextureToTexture => {
            // Initialize the source, then copy a single slice into the destination at z=1.
            let src_tex = create_3d_texture(ctx, "3d dest init source texture", 1);
            ctx.queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &src_tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &vec![SENTINEL; slice_bytes],
                d3_buffer_layout(),
                one_slice,
            );
            ctx.queue.submit(None);
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo {
                    texture: &src_tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                dst_info,
                one_slice,
            );
        }
        WriteMethod::WriteTexture => {
            unreachable!("3D dest-init tests exercise only the encoder copy commands")
        }
    }
    // Submit the partial copy on its own to ensure the init action is applied on its own,
    // and not in combination with an init action for the readback.
    ctx.queue.submit(Some(encoder.finish()));

    // The whole texture is read back from the origin, so route it through
    // `ReadbackBuffers`. The written slice (z = DST_Z) must keep its sentinel data
    // and every untouched slice must be zero-initialized.
    let readback_buffers = ReadbackBuffers::new(&ctx.device, &dst);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    readback_buffers.copy_from(&ctx.device, &mut encoder, &dst);
    ctx.queue.submit(Some(encoder.finish()));

    let mut expected = vec![0u8; slice_bytes * D3_DEPTH as usize];
    let written_start = DST_Z as usize * slice_bytes;
    expected[written_start..written_start + slice_bytes].fill(SENTINEL);
    readback_buffers
        .assert_buffer_contents(ctx, &expected)
        .await;
}
