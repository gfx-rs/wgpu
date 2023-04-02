use crate::common::{image::ReadbackBuffers, initialize_test, TestParameters, TestingContext};
use wasm_bindgen_test::*;
use wgpu::*;

// Checks if discarding a color target resets its init state, causing a zero read of this texture when copied in after submit of the encoder.
#[test]
#[wasm_bindgen_test]
fn discarding_color_target_resets_texture_init_state_check_visible_on_copy_after_submit() {
    initialize_test(TestParameters::default().webgl2_failure(), |mut ctx| {
        let mut case = TestCase::new(&mut ctx, TextureFormat::Rgba8UnormSrgb);
        case.create_command_encoder();
        case.discard();
        case.submit_command_encoder();

        case.create_command_encoder();
        case.copy_texture_to_buffer();
        case.submit_command_encoder();

        case.assert_buffers_are_zero();
    });
}

// Checks if discarding a color target resets its init state, causing a zero read of this texture when copied in the same encoder to a buffer.
#[test]
#[wasm_bindgen_test]
fn discarding_color_target_resets_texture_init_state_check_visible_on_copy_in_same_encoder() {
    initialize_test(TestParameters::default().webgl2_failure(), |mut ctx| {
        let mut case = TestCase::new(&mut ctx, TextureFormat::Rgba8UnormSrgb);
        case.create_command_encoder();
        case.discard();
        case.copy_texture_to_buffer();
        case.submit_command_encoder();

        case.assert_buffers_are_zero();
    });
}

#[test]
#[wasm_bindgen_test]
fn discarding_depth_target_resets_texture_init_state_check_visible_on_copy_in_same_encoder() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(
                DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES | DownlevelFlags::COMPUTE_SHADERS,
            )
            .limits(Limits::downlevel_defaults())
            .specific_failure(
                Some(wgpu::Backends::DX12),
                Some(5140),
                Some("Microsoft Basic Render Driver"),
                false,
            ),
        |mut ctx| {
            for format in [
                TextureFormat::Stencil8,
                TextureFormat::Depth16Unorm,
                TextureFormat::Depth24Plus,
                TextureFormat::Depth24PlusStencil8,
                TextureFormat::Depth32Float,
            ] {
                let mut case = TestCase::new(&mut ctx, format);
                case.create_command_encoder();
                case.discard();
                case.copy_texture_to_buffer();
                case.submit_command_encoder();

                case.assert_buffers_are_zero();
            }
        },
    );
}

#[test]
#[wasm_bindgen_test]
fn discarding_either_depth_or_stencil_aspect() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(
                DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES | DownlevelFlags::COMPUTE_SHADERS,
            )
            .limits(Limits::downlevel_defaults()),
        |mut ctx| {
            let mut case = TestCase::new(&mut ctx, TextureFormat::Depth24PlusStencil8);
            case.create_command_encoder();
            case.discard_depth();
            case.submit_command_encoder();

            case.create_command_encoder();
            case.discard_stencil();
            case.submit_command_encoder();

            case.create_command_encoder();
            case.copy_texture_to_buffer();
            case.submit_command_encoder();

            case.assert_buffers_are_zero();
        },
    );
}

struct TestCase<'ctx> {
    ctx: &'ctx mut TestingContext,
    format: TextureFormat,
    texture: Texture,
    readback_buffers: ReadbackBuffers,
    encoder: Option<CommandEncoder>,
}

impl<'ctx> TestCase<'ctx> {
    pub fn new(ctx: &'ctx mut TestingContext, format: TextureFormat) -> Self {
        let extra_usages = match format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                TextureUsages::TEXTURE_BINDING
            }
            _ => TextureUsages::empty(),
        };

        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("RenderTarget"),
            size: Extent3d {
                width: COPY_BYTES_PER_ROW_ALIGNMENT,
                height: COPY_BYTES_PER_ROW_ALIGNMENT,
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
                        store: true,
                    }),
                    stencil_ops: format.has_stencil_aspect().then_some(Operations {
                        load: LoadOp::Clear(0xFFFFFFFF),
                        store: true,
                    }),
                }),
                timestamp_writes: &[],
            });
            ctx.queue.submit([encoder.finish()]);
        } else {
            let block_size = format.block_size(None).unwrap();
            let bytes_per_row = texture.width() * block_size;

            // Size for tests is chosen so that we don't need to care about buffer alignments.
            assert!(!format.is_compressed());
            assert_eq!(bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);

            let buffer_size = texture.height() * bytes_per_row;
            let data = vec![255; buffer_size as usize];
            ctx.queue.write_texture(
                ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                &data,
                ImageDataLayout {
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
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: false, // discard!
                        },
                    },
                )],
                depth_stencil_attachment: self.format.is_depth_stencil_format().then_some(
                    RenderPassDepthStencilAttachment {
                        view: &self.texture.create_view(&TextureViewDescriptor::default()),
                        depth_ops: self.format.has_depth_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: false, // discard!
                        }),
                        stencil_ops: self.format.has_stencil_aspect().then_some(Operations {
                            load: LoadOp::Load,
                            store: false, // discard!
                        }),
                    },
                ),
                timestamp_writes: &[],
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
                        depth_ops: Some(Operations {
                            load: LoadOp::Load,
                            store: false, // discard!
                        }),
                        stencil_ops: self.format.has_stencil_aspect().then_some(Operations {
                            load: LoadOp::Clear(0),
                            store: true,
                        }),
                    },
                ),
                timestamp_writes: &[],
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
                            store: true,
                        }),
                        stencil_ops: Some(Operations {
                            load: LoadOp::Load,
                            store: false, // discard!
                        }),
                    },
                ),
                timestamp_writes: &[],
            });
    }

    pub fn copy_texture_to_buffer(&mut self) {
        self.readback_buffers.copy_from(
            &self.ctx.device,
            self.encoder.as_mut().unwrap(),
            &self.texture,
        );
    }

    pub fn assert_buffers_are_zero(&mut self) {
        assert!(
            self.readback_buffers.are_zero(&self.ctx.device),
            "texture was not fully cleared"
        );
    }
}
