//! Tests for zero-initialization of resources.
//!
//! It is common for allocations on a fresh heap to coincidentally be zero, which can cause
//! these tests to produce false negatives. One way to make them more reliable is to run
//! them on llvmpipe with `LVP_POISON_MEMORY=true` in the environment.

use core::num::NonZeroU64;

use wgpu::util::DeviceExt as _;
use wgpu::*;
use wgpu_test::{
    apply, gpu_test,
    image::{ReadbackBuffers, ReadbackData},
    FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
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
        DISCARDING_3D_DEPTH_SLICE_ALONGSIDE_ANOTHER_SLICE_IN_SAME_PASS,
        DISCARDING_3D_DEPTH_SLICE_PRESERVES_OTHER_SLICES,
        DISCARDING_BOTH_DEPTH_AND_STENCIL_WITH_DIVERGING_LOAD_OPS,
        DISCARDING_COLOR_TARGET_AT_NONZERO_MIP_AND_LAYER,
        DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE,
        DISCARDING_COLOR_TARGET_THEN_LOADING_IT_IN_SAME_ENCODER,
        DISCARDING_DEPTH_TARGET_AT_NONZERO_MIP_AND_LAYER,
        DISCARDING_DEPTH_TARGET_AT_NONZERO_MIP_AND_LAYER_DEPTH32FLOAT_STENCIL8,
        DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE,
        DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE_DEPTH32FLOAT_STENCIL8,
        DISCARDING_EITHER_DEPTH_OR_STENCIL_ASPECT_AT_NONZERO_MIP_AND_LAYER,
        DISCARDING_EITHER_DEPTH_OR_STENCIL_ASPECT_TEST,
        RENDER_PASS_LOAD_INITS_WHOLE_3D_MIP,
        RENDER_PASS_STORE_INITS_ONLY_TARGET_3D_MIP,
        RENDER_PASS_STORE_INITS_ONLY_TARGET_COLOR_SUBRESOURCE,
        RENDER_PASS_STORE_INITS_ONLY_TARGET_DEPTH_STENCIL_SUBRESOURCE,
        RENDER_PASS_STORE_TO_3D_DEPTH_SLICE_INITS_OTHER_SLICES,
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
        VERTEX_BUFFER_TAIL_INIT_PLAIN,
        VERTEX_BUFFER_TAIL_INIT_MAP_WRITE,
        VERTEX_BUFFER_TAIL_INIT_MAPPED_AT_CREATION,
        VERTEX_BUFFER_TAIL_INIT_MAP_WRITE_MAPPED_AT_CREATION,
        COPY_TEXTURE_TO_BUFFER_UNALIGNED_OFFSET_ROW_PADDING_INIT,
        COPY_TEXTURE_TO_BUFFER_UNALIGNED_OFFSET_IMAGE_PADDING_INIT,
        MARK_EXTERNALLY_INITIALIZED_SKIPS_LAZY_CLEAR,
    ]);
}

// Checks if discarding a color target resets its init state, causing a zero read of this texture
// when copied in the same encoder as the discarding pass, or in a later one.
#[apply(gpu_test!)]
static DISCARDING_COLOR_TARGET_RESETS_TEXTURE_INIT_STATE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        // https://github.com/gfx-rs/wgpu/issues/10162
        .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
        .run_async(|ctx| async move {
            check_discard_resets_whole_texture(&ctx, TextureFormat::Rgba8UnormSrgb).await;
        });

// As above, for depth/stencil targets.
#[apply(gpu_test!)]
static DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE: GpuTestConfiguration =
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
            for &format in CORE_DEPTH_STENCIL_FORMATS {
                check_discard_resets_whole_texture(&ctx, format).await;
            }
        });

// As above, for the depth/stencil format that is behind an optional feature.
#[apply(gpu_test!)]
static DISCARDING_DEPTH_TARGET_RESETS_TEXTURE_INIT_STATE_DEPTH32FLOAT_STENCIL8:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::DEPTH32FLOAT_STENCIL8)
            .downlevel_flags(DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        check_discard_resets_whole_texture(&ctx, TextureFormat::Depth32FloatStencil8).await;
    });

#[apply(gpu_test!)]
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
        .run_async(|ctx| async move {
            for &format in CORE_DEPTH_STENCIL_FORMATS {
                let mut case = discard_case(&ctx, format, false);
                case.create_command_encoder();
                case.pass(
                    RenderTarget::Array { mip: 0, layer: 0 },
                    PassOps::discard_depth_keep_stencil(),
                );
                case.submit_command_encoder();

                case.create_command_encoder();
                case.pass(
                    RenderTarget::Array { mip: 0, layer: 0 },
                    PassOps::discard_stencil_keep_depth(),
                );
                case.submit_command_encoder();

                case.create_command_encoder();
                case.copy_texture_to_buffer();
                case.submit_command_encoder();

                case.assert_buffers_are_zero().await;
            }
        });

/// A single-mip, single-layer render target, as used by the
/// `DISCARDING_*_RESETS_TEXTURE_INIT_STATE_*` tests.
fn discard_case(
    ctx: &TestingContext,
    format: TextureFormat,
    readback_in_same_encoder: bool,
) -> RenderTargetInitCase<'_> {
    RenderTargetInitCase::new(
        ctx,
        TextureSpec {
            format,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: SUBRESOURCE_WIDTH,
                height: SUBRESOURCE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
        },
        TextureViewDimension::D2,
        PreInit::Sentinel,
        format!(
            "{format:?} target discard{}",
            if readback_in_same_encoder {
                ", read back in the same encoder"
            } else {
                ""
            }
        ),
    )
}

/// Discards the only subresource of a `format` texture and checks that the whole texture
/// reads back as zero, both when the readback copy is in the same encoder as the discarding
/// pass and when it is in a later one.
async fn check_discard_resets_whole_texture(ctx: &TestingContext, format: TextureFormat) {
    for readback_in_same_encoder in [false, true] {
        let mut case = discard_case(ctx, format, readback_in_same_encoder);
        case.create_command_encoder();
        case.pass(
            RenderTarget::Array { mip: 0, layer: 0 },
            PassOps::load_discard(),
        );
        if !readback_in_same_encoder {
            case.submit_command_encoder();
            case.create_command_encoder();
        }
        case.copy_texture_to_buffer();
        case.submit_command_encoder();

        case.assert_buffers_are_zero().await;
    }
}

/// The texture used by a render-target initialization test case.
#[derive(Clone, Copy)]
struct TextureSpec {
    format: TextureFormat,
    /// Either `D2` or `D3`.
    dimension: TextureDimension,
    size: Extent3d,
    mip_level_count: u32,
}

impl TextureSpec {
    fn mip_size(&self, mip_level: u32) -> Extent3d {
        self.size.mip_level_size(mip_level, self.dimension)
    }

    /// Number of array layers (`D2`) or depth slices (`D3`) at `mip_level`.
    fn subresources_at(&self, mip_level: u32) -> u32 {
        self.mip_size(mip_level).depth_or_array_layers
    }

    /// The aspects whose exact contents are checked. [`ReadbackBuffers`] reads a combined
    /// depth/stencil format into one buffer per aspect, and both of them are checked.
    fn readback_aspects(&self) -> &'static [TextureAspect] {
        match self.format {
            TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 => {
                &[TextureAspect::DepthOnly, TextureAspect::StencilOnly]
            }
            TextureFormat::Depth24Plus => &[TextureAspect::DepthOnly],
            _ => &[TextureAspect::All],
        }
    }

    fn texel_size(&self, aspect: TextureAspect) -> u32 {
        match (self.format, aspect) {
            // Read back by a compute shader, which writes one `f32` per texel.
            (
                TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8,
                TextureAspect::DepthOnly,
            ) => 4,
            _ => self
                .format
                .block_copy_size(Some(aspect))
                .expect("aspect is not directly readable"),
        }
    }

    /// Row stride of a tightly packed image of `mip_level`, for `write_texture`, which
    /// imposes no row alignment requirement. Only used for the single-aspect color formats.
    fn bytes_per_row(&self, mip_level: u32) -> u32 {
        self.mip_size(mip_level).width * self.texel_size(TextureAspect::All)
    }

    fn create(&self, ctx: &TestingContext) -> Texture {
        // `ReadbackBuffers` reads the depth aspect of these formats with a compute shader.
        let extra_usages = match self.format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                TextureUsages::TEXTURE_BINDING
            }
            _ => TextureUsages::empty(),
        };

        assert!(!self.format.is_compressed());

        ctx.device.create_texture(&TextureDescriptor {
            label: Some("RenderTarget"),
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: self.dimension,
            format: self.format,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT
                | extra_usages,
            view_formats: &[],
        })
    }
}

/// The single subresource that the render pass under test targets.
#[derive(Clone, Copy)]
enum RenderTarget {
    /// A (mip level, array layer) of a 2D texture.
    Array { mip: u32, layer: u32 },
    /// A (mip level, depth slice) of a 3D texture.
    Volume { mip: u32, depth_slice: u32 },
}

impl RenderTarget {
    fn mip(&self) -> u32 {
        match *self {
            Self::Array { mip, .. } | Self::Volume { mip, .. } => mip,
        }
    }

    /// Index of this target within the readback of its mip level.
    fn subresource_index(&self) -> u32 {
        match *self {
            Self::Array { layer, .. } => layer,
            Self::Volume { depth_slice, .. } => depth_slice,
        }
    }
}

impl core::fmt::Display for RenderTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Array { mip, layer } => write!(f, "mip {mip}, layer {layer}"),
            Self::Volume { mip, depth_slice } => write!(f, "mip {mip}, slice {depth_slice}"),
        }
    }
}

/// Operations for the render pass under test. Depth and stencil are kept separate so that
/// passes with diverging depth/stencil operations can be constructed.
#[derive(Clone, Copy)]
struct PassOps {
    color: Operations<Color>,
    depth: Operations<f32>,
    stencil: Operations<u32>,
}

impl PassOps {
    /// Write the sentinel value into the target.
    fn clear_store(sentinel: Sentinel) -> Self {
        Self {
            color: Operations {
                load: LoadOp::Clear(sentinel.color),
                store: StoreOp::Store,
            },
            depth: Operations {
                load: LoadOp::Clear(sentinel.depth),
                store: StoreOp::Store,
            },
            stencil: Operations {
                load: LoadOp::Clear(sentinel.stencil),
                store: StoreOp::Store,
            },
        }
    }

    /// Clear the target to the sentinel value, then discard it.
    fn clear_discard(sentinel: Sentinel) -> Self {
        let mut ops = Self::clear_store(sentinel);
        ops.color.store = StoreOp::Discard;
        ops.depth.store = StoreOp::Discard;
        ops.stencil.store = StoreOp::Discard;
        ops
    }

    fn load_store() -> Self {
        Self {
            color: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
            depth: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
            stencil: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        }
    }

    fn load_discard() -> Self {
        let mut ops = Self::load_store();
        ops.color.store = StoreOp::Discard;
        ops.depth.store = StoreOp::Discard;
        ops.stencil.store = StoreOp::Discard;
        ops
    }

    /// Discard the depth aspect while storing a zeroed stencil aspect.
    fn discard_depth_keep_stencil() -> Self {
        let mut ops = Self::load_discard();
        ops.stencil = Operations {
            load: LoadOp::Clear(0),
            store: StoreOp::Store,
        };
        ops
    }

    /// Discard the stencil aspect while storing a zeroed depth aspect.
    fn discard_stencil_keep_depth() -> Self {
        let mut ops = Self::load_discard();
        ops.depth = Operations {
            load: LoadOp::Clear(0.0),
            store: StoreOp::Store,
        };
        ops
    }

    /// Discard both aspects, loading only one of them and clearing the other to zero, so
    /// that the load operations diverge while the store operations match.
    fn discard_both_with_diverging_load(load_depth: bool) -> Self {
        let mut ops = Self::load_discard();
        if load_depth {
            ops.stencil.load = LoadOp::Clear(0);
        } else {
            ops.depth.load = LoadOp::Clear(0.0);
        }
        ops
    }
}

/// A non-zero value written into a render target, and the bytes it must read back as.
#[derive(Clone, Copy)]
struct Sentinel {
    color: Color,
    depth: f32,
    stencil: u32,
    /// Bytes that the color aspect, or the depth aspect of a depth/stencil format, must read
    /// back as.
    texel: [u8; 4],
    texel_size: usize,
    /// Byte that the stencil aspect must read back as, for a format that has one that is
    /// read separately.
    stencil_texel: [u8; 1],
}

impl Sentinel {
    /// Bytes that one texel of `aspect` must read back as.
    fn texel(&self, aspect: TextureAspect) -> &[u8] {
        match aspect {
            TextureAspect::StencilOnly => &self.stencil_texel,
            _ => &self.texel[..self.texel_size],
        }
    }
}

/// Clear values are chosen to be exactly representable in the target format, and non-zero,
/// so that a zero-filled subresource is distinguishable from the sentinel.
///
/// Most are also distinct from the byte that [`ReadbackBuffers`] poisons its buffers with, so
/// that a readback that never happened is distinguishable too. [`TextureFormat::Rgba8UnormSrgb`]
/// is the exception: 1.0 is the only non-zero value that the linear to sRGB clear value
/// conversion is guaranteed to reproduce exactly, and it is all-ones, like the poison byte.
/// That is harmless, because the tests that use the format only assert that the texture reads
/// back as zero.
fn sentinel_for(format: TextureFormat) -> Sentinel {
    // 0.75 * 65535 = 49151.25 is far enough from a rounding boundary that any
    // implementation produces 49151 (0xBFFF) for `Depth16Unorm`.
    let mut depth = 0.75;
    let stencil = 0xAB;

    let (color, texel, texel_size) = match format {
        // Integer clear values are not subject to any conversion.
        TextureFormat::Rgba8Uint => (
            Color {
                r: f64::from(0x11_u8),
                g: f64::from(0x22_u8),
                b: f64::from(0x33_u8),
                a: f64::from(0x44_u8),
            },
            [0x11, 0x22, 0x33, 0x44],
            4,
        ),
        TextureFormat::Rgba8UnormSrgb => (Color::WHITE, [0xFF; 4], 4),
        TextureFormat::Stencil8 => (Color::TRANSPARENT, [stencil as u8, 0, 0, 0], 1),
        TextureFormat::Depth16Unorm => (Color::TRANSPARENT, [0xFF, 0xBF, 0, 0], 2),
        TextureFormat::Depth32Float | TextureFormat::Depth32FloatStencil8 => {
            (Color::TRANSPARENT, 0.75_f32.to_le_bytes(), 4)
        }
        TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
            // The depth aspect of these formats is read back as `f32` by a compute shader,
            // and a backend may give it either 24 bit unorm or `f32` precision. 1.0 is the
            // only non-zero value that both of those represent exactly.
            depth = 1.0;
            (Color::TRANSPARENT, 1.0_f32.to_le_bytes(), 4)
        }
        _ => unimplemented!("no sentinel value defined for {format:?}"),
    };

    Sentinel {
        color,
        depth,
        stencil,
        texel,
        texel_size,
        stencil_texel: [stencil as u8],
    }
}

/// Expected contents of a single mip level / array layer / depth slice.
#[derive(Clone, Copy, Debug)]
enum Expected {
    Zero,
    Sentinel,
}

/// Contents of the texture before the render pass under test.
#[derive(Clone, Copy)]
enum PreInit {
    /// Leave the texture untouched, i.e. uninitialized.
    Fresh,
    /// Write the sentinel value into every mip level and array layer / depth slice.
    Sentinel,
}

struct RenderTargetInitCase<'ctx> {
    ctx: &'ctx TestingContext,
    spec: TextureSpec,
    sentinel: Sentinel,
    view_dimension: TextureViewDimension,
    texture: Texture,
    readback: ReadbackBuffers,
    encoder: Option<CommandEncoder>,
    /// Prefix of every assertion message, identifying the case.
    desc: String,
}

impl<'ctx> RenderTargetInitCase<'ctx> {
    fn new(
        ctx: &'ctx TestingContext,
        spec: TextureSpec,
        view_dimension: TextureViewDimension,
        pre_init: PreInit,
        desc: String,
    ) -> Self {
        let texture = spec.create(ctx);
        let sentinel = sentinel_for(spec.format);
        let readback = ReadbackBuffers::new(&ctx.device, &texture);

        let case = Self {
            ctx,
            spec,
            sentinel,
            view_dimension,
            texture,
            readback,
            encoder: None,
            desc,
        };
        if matches!(pre_init, PreInit::Sentinel) {
            case.write_sentinel();
        }
        case
    }

    /// Fill every mip level and array layer / depth slice with the sentinel value.
    fn write_sentinel(&self) {
        // Color textures are filled with `write_texture` rather than a render pass clear,
        // to maximize the chance that these tests detect a missing zero-init. On at least
        // one Vulkan driver, what the discard pass leaves behind depends on how the texture
        // is filled here:
        // * clear to white -> the discarded texture reads back as black, which is what the
        //   test asserts, so a wgpu that never zero-initialized any discarded texture would
        //   still pass
        // * clear to red -> the discarded texture stays red
        // * write_texture -> the discarded texture keeps its contents
        // None of that violates any spec: it is wgpu's job to zero-fill the texture, no
        // matter what the discard does or does not do.
        //
        // Depth/stencil textures have to use a clear regardless, because their depth aspect
        // cannot be the destination of a `write_texture`, so they remain exposed to this
        // possibility of a false-negative.
        if self.spec.format.is_depth_stencil_format() {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Depth/Stencil setup"),
                });
            for mip_level in 0..self.spec.mip_level_count {
                for layer in 0..self.spec.subresources_at(mip_level) {
                    self.record_pass(
                        &mut encoder,
                        RenderTarget::Array {
                            mip: mip_level,
                            layer,
                        },
                        PassOps::clear_store(self.sentinel),
                    );
                }
            }
            self.ctx.queue.submit([encoder.finish()]);
        } else {
            for mip_level in 0..self.spec.mip_level_count {
                let mip_size = self.spec.mip_size(mip_level);
                let bytes_per_row = self.spec.bytes_per_row(mip_level);
                let texels = mip_size.height * mip_size.depth_or_array_layers * mip_size.width;
                let texel = self.sentinel.texel(TextureAspect::All);
                let data: Vec<u8> = texel
                    .iter()
                    .copied()
                    .cycle()
                    .take(texels as usize * texel.len())
                    .collect();
                self.ctx.queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(mip_size.height),
                    },
                    mip_size,
                );
            }
            self.ctx.queue.submit(None);
        }
    }

    /// Record a render pass whose only attachment is `target`. The pass has no pipeline and
    /// issues no draws; its load and store operations are the entire point.
    fn record_pass(&self, encoder: &mut CommandEncoder, target: RenderTarget, ops: PassOps) {
        let format = self.spec.format;
        let (base_array_layer, depth_slice) = match target {
            RenderTarget::Array { layer, .. } => (layer, None),
            RenderTarget::Volume { depth_slice, .. } => (0, Some(depth_slice)),
        };
        let view = self.texture.create_view(&TextureViewDescriptor {
            label: None,
            format: Some(format),
            dimension: Some(self.view_dimension),
            usage: Some(TextureUsages::RENDER_ATTACHMENT),
            aspect: TextureAspect::All,
            base_mip_level: target.mip(),
            mip_level_count: Some(1),
            base_array_layer,
            array_layer_count: Some(1),
            swizzle: Default::default(),
        });
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render target under test"),
            color_attachments: &[format
                .has_color_aspect()
                .then_some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice,
                    resolve_target: None,
                    ops: ops.color,
                })],
            depth_stencil_attachment: format.is_depth_stencil_format().then_some(
                RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: format.has_depth_aspect().then_some(ops.depth),
                    stencil_ops: format.has_stencil_aspect().then_some(ops.stencil),
                },
            ),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    /// Record a render pass with one color attachment per entry of `targets`, which must all
    /// be distinct depth slices of the same mip level of a 3D texture.
    fn record_multi_slice_pass(
        &self,
        encoder: &mut CommandEncoder,
        targets: &[(RenderTarget, PassOps)],
    ) {
        let views: Vec<TextureView> = targets
            .iter()
            .map(|(target, _)| {
                self.texture.create_view(&TextureViewDescriptor {
                    label: None,
                    format: Some(self.spec.format),
                    dimension: Some(self.view_dimension),
                    usage: Some(TextureUsages::RENDER_ATTACHMENT),
                    aspect: TextureAspect::All,
                    base_mip_level: target.mip(),
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    swizzle: Default::default(),
                })
            })
            .collect();
        let attachments: Vec<Option<RenderPassColorAttachment<'_>>> = targets
            .iter()
            .zip(&views)
            .map(|((target, ops), view)| {
                Some(RenderPassColorAttachment {
                    view,
                    depth_slice: Some(target.subresource_index()),
                    resolve_target: None,
                    ops: ops.color,
                })
            })
            .collect();
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render targets under test"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn create_command_encoder(&mut self) {
        self.encoder = Some(
            self.ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default()),
        )
    }

    fn submit_command_encoder(&mut self) {
        self.ctx
            .queue
            .submit([self.encoder.take().unwrap().finish()]);
    }

    fn pass(&mut self, target: RenderTarget, ops: PassOps) {
        let mut encoder = self.encoder.take().unwrap();
        self.record_pass(&mut encoder, target, ops);
        self.encoder = Some(encoder);
    }

    fn multi_slice_pass(&mut self, targets: &[(RenderTarget, PassOps)]) {
        let mut encoder = self.encoder.take().unwrap();
        self.record_multi_slice_pass(&mut encoder, targets);
        self.encoder = Some(encoder);
    }

    fn copy_texture_to_buffer(&mut self) {
        let mut encoder = self.encoder.take().unwrap();
        self.readback
            .copy_from(&self.ctx.device, &mut encoder, &self.texture);
        self.encoder = Some(encoder);
    }

    /// Compare every texel of every mip level and array layer / depth slice, of every aspect
    /// that [`TextureSpec::readback_aspects`] selects, against `expected`, which is called
    /// with (mip level, array layer or depth slice).
    async fn assert_contents(&self, expected: impl Fn(u32, u32) -> Expected) {
        let aspects = self.spec.readback_aspects();
        let mut failure = None;

        for &aspect in aspects {
            // Naming the aspect is noise for a format that has only one.
            let aspect_note = match aspect {
                _ if aspects.len() == 1 => "",
                TextureAspect::DepthOnly => ", depth aspect",
                _ => ", stencil aspect",
            };
            let data = self.readback.retrieve(self.ctx, aspect).await;
            let aspect_failure = self.check_aspect(aspect, aspect_note, &data, &expected);
            if failure.is_none() {
                failure = aspect_failure;
            }
        }

        if let Some(failure) = failure {
            panic!("{failure}");
        }
    }

    /// Check one aspect of a readback, logging the result for every subresource and
    /// returning a message describing the first mismatch, if there was one.
    fn check_aspect(
        &self,
        aspect: TextureAspect,
        aspect_note: &str,
        data: &ReadbackData,
        expected: &impl Fn(u32, u32) -> Expected,
    ) -> Option<String> {
        let texel_size = self.spec.texel_size(aspect) as usize;
        assert_eq!(texel_size, self.sentinel.texel(aspect).len());
        let case_desc = &self.desc;
        let subresource_name = match self.spec.dimension {
            TextureDimension::D3 => "slice",
            _ => "layer",
        };
        let zero = [0_u8; 4];
        let mut failure = None;

        for mip_level in 0..self.spec.mip_level_count {
            let width = self.spec.mip_size(mip_level).width;

            for index in 0..self.spec.subresources_at(mip_level) {
                let expected = expected(mip_level, index);
                let want = match expected {
                    Expected::Zero => &zero[..texel_size],
                    Expected::Sentinel => self.sentinel.texel(aspect),
                };

                let texels = data.subresource(mip_level, index).chunks_exact(texel_size);
                let texel_count = texels.len();
                let mut wrong = 0;
                let mut first_wrong = None;
                for (i, texel) in texels.enumerate() {
                    if texel != want {
                        wrong += 1;
                        if first_wrong.is_none() {
                            first_wrong = Some((i, texel));
                        }
                    }
                }

                if wrong == 0 {
                    log::info!(
                        "{case_desc}: mip {mip_level}, {subresource_name} {index}{aspect_note}: ok"
                    );
                    continue;
                }
                log::info!(
                    "{case_desc}: mip {mip_level}, {subresource_name} {index}{aspect_note}: \
                     {wrong}/{texel_count} texels wrong"
                );

                let (i, texel) = first_wrong.unwrap();
                if failure.is_none() {
                    let problem = match expected {
                        Expected::Zero => "read back non-zero",
                        Expected::Sentinel => "lost its contents",
                    };
                    failure = Some(format!(
                        "{case_desc}: mip {mip_level}, {subresource_name} \
                         {index}{aspect_note} {problem} at x={x} y={y}: \
                         expected {want:02x?}, got {texel:02x?}",
                        x = i as u32 % width,
                        y = i as u32 / width,
                    ));
                }
            }
        }

        failure
    }

    /// Assert that `target` holds `target_expected` and every other subresource holds the
    /// opposite.
    async fn assert_only_target(&self, target: RenderTarget, target_expected: Expected) {
        let other = match target_expected {
            Expected::Zero => Expected::Sentinel,
            Expected::Sentinel => Expected::Zero,
        };
        self.assert_contents(|mip_level, index| {
            if mip_level == target.mip() && index == target.subresource_index() {
                target_expected
            } else {
                other
            }
        })
        .await
    }

    async fn assert_buffers_are_zero(&self) {
        self.assert_contents(|_, _| Expected::Zero).await
    }
}

// Tests of initialization tracking for textures used as render targets.
//
// Init tracking is per (mip level, array layer), and a render pass attachment is always a
// view of a single mip level and a single array layer. Depth slices of 3D textures are not
// tracked at all: for a 3D texture, one mip level is a single tracking unit covering all of
// its slices.

/// Mip level count and array layer count of the textures used by the subresource tests.
/// Every combination of a mip level and an array layer is a distinct tracking unit.
const SUBRESOURCE_MIPS: u32 = 3;
const SUBRESOURCE_LAYERS: u32 = 3;
/// Size of mip level 0 of the textures used by these tests. Non-square, so a width/height
/// swap is caught. Deliberately small: the depth aspect of `Depth24Plus` and
/// `Depth24PlusStencil8` is read back by a single-invocation compute shader in
/// [`ReadbackBuffers`].
const SUBRESOURCE_WIDTH: u32 = 256;
const SUBRESOURCE_HEIGHT: u32 = 4;

fn color_2d_spec() -> TextureSpec {
    TextureSpec {
        format: TextureFormat::Rgba8Uint,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: SUBRESOURCE_WIDTH,
            height: SUBRESOURCE_HEIGHT,
            depth_or_array_layers: SUBRESOURCE_LAYERS,
        },
        mip_level_count: SUBRESOURCE_MIPS,
    }
}

/// Mip level 0 has four depth slices and mip level 1 has two, so a single-slice attachment
/// covers only part of the tracking unit. Mip level 2 has a single slice, which one
/// attachment covers completely.
fn color_3d_spec() -> TextureSpec {
    TextureSpec {
        format: TextureFormat::Rgba8Uint,
        dimension: TextureDimension::D3,
        size: Extent3d {
            width: SUBRESOURCE_WIDTH,
            height: SUBRESOURCE_HEIGHT,
            depth_or_array_layers: 4,
        },
        mip_level_count: SUBRESOURCE_MIPS,
    }
}

/// Depth/stencil formats other than [`TextureFormat::Depth32FloatStencil8`],
/// which requires [`Features::DEPTH32FLOAT_STENCIL8`].
const CORE_DEPTH_STENCIL_FORMATS: &[TextureFormat] = &[
    TextureFormat::Stencil8,
    TextureFormat::Depth16Unorm,
    TextureFormat::Depth24Plus,
    TextureFormat::Depth24PlusStencil8,
    TextureFormat::Depth32Float,
];

fn depth_stencil_2d_spec(format: TextureFormat) -> TextureSpec {
    TextureSpec {
        format,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: SUBRESOURCE_WIDTH,
            height: SUBRESOURCE_HEIGHT,
            depth_or_array_layers: SUBRESOURCE_LAYERS,
        },
        mip_level_count: SUBRESOURCE_MIPS,
    }
}

/// Renders the sentinel value into `target` with `StoreOp::Store` and checks that only
/// `target` holds it, and that every other subresource of the texture reads back as zero.
///
/// The sentinel check fails deterministically if the store is tracked as initializing too
/// little; the zero checks only fail if the underlying allocation is not already zero (see
/// the comment at the top of this file).
async fn check_store_inits_only_target(
    ctx: &TestingContext,
    spec: TextureSpec,
    view_dimension: TextureViewDimension,
    target: RenderTarget,
    readback_in_same_encoder: bool,
) {
    let mut case = RenderTargetInitCase::new(
        ctx,
        spec,
        view_dimension,
        PreInit::Fresh,
        format!(
            "{:?} store to {target} of a {view_dimension:?} view{}",
            spec.format,
            if readback_in_same_encoder {
                ", read back in the same encoder"
            } else {
                ""
            }
        ),
    );

    let ops = PassOps::clear_store(case.sentinel);
    case.create_command_encoder();
    case.pass(target, ops);
    if !readback_in_same_encoder {
        case.submit_command_encoder();
        case.create_command_encoder();
    }
    case.copy_texture_to_buffer();
    case.submit_command_encoder();

    case.assert_only_target(target, Expected::Sentinel).await;
}

/// Discards `target` and checks that only `target` was reset to zero, and that every other
/// subresource of the texture kept the sentinel value.
async fn check_discard_resets_only_target(
    ctx: &TestingContext,
    spec: TextureSpec,
    view_dimension: TextureViewDimension,
    target: RenderTarget,
    load: LoadOp<()>,
    readback_in_same_encoder: bool,
) {
    let mut case = RenderTargetInitCase::new(
        ctx,
        spec,
        view_dimension,
        PreInit::Sentinel,
        format!(
            "{:?} {} + discard of {target}{}",
            spec.format,
            if matches!(load, LoadOp::Load) {
                "load"
            } else {
                "clear"
            },
            if readback_in_same_encoder {
                ", read back in the same encoder"
            } else {
                ""
            }
        ),
    );

    let ops = if matches!(load, LoadOp::Load) {
        PassOps::load_discard()
    } else {
        PassOps::clear_discard(case.sentinel)
    };
    case.create_command_encoder();
    case.pass(target, ops);
    if !readback_in_same_encoder {
        case.submit_command_encoder();
        case.create_command_encoder();
    }
    case.copy_texture_to_buffer();
    case.submit_command_encoder();

    case.assert_only_target(target, Expected::Zero).await;
}

/// Discards a subresource of a mipmapped, layered `format` texture that is neither the first
/// mip level nor the first array layer, and checks that no other subresource was affected.
async fn check_depth_discard_at_nonzero_mip_and_layer(ctx: &TestingContext, format: TextureFormat) {
    for target in [
        RenderTarget::Array { mip: 1, layer: 1 },
        RenderTarget::Array { mip: 2, layer: 2 },
    ] {
        for readback_in_same_encoder in [false, true] {
            check_discard_resets_only_target(
                ctx,
                depth_stencil_2d_spec(format),
                TextureViewDimension::D2,
                target,
                LoadOp::Load,
                readback_in_same_encoder,
            )
            .await;
        }
    }
}

// A `StoreOp::Store` pass must mark only the (mip level, array layer) that it targets as
// initialized.
#[apply(gpu_test!)]
static RENDER_PASS_STORE_INITS_ONLY_TARGET_COLOR_SUBRESOURCE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default())
        .run_async(|ctx| async move {
            for view_dimension in [TextureViewDimension::D2, TextureViewDimension::D2Array] {
                for target in [
                    RenderTarget::Array { mip: 0, layer: 0 },
                    RenderTarget::Array { mip: 1, layer: 1 },
                    RenderTarget::Array { mip: 2, layer: 2 },
                    RenderTarget::Array { mip: 2, layer: 0 },
                ] {
                    for readback_in_same_encoder in [false, true] {
                        check_store_inits_only_target(
                            &ctx,
                            color_2d_spec(),
                            view_dimension,
                            target,
                            readback_in_same_encoder,
                        )
                        .await;
                    }
                }
            }
        });

#[apply(gpu_test!)]
static RENDER_PASS_STORE_INITS_ONLY_TARGET_DEPTH_STENCIL_SUBRESOURCE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // `ReadbackBuffers` reads the depth aspect of `Depth24PlusStencil8` with a
                // compute shader.
                .downlevel_flags(
                    DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES
                        | DownlevelFlags::COMPUTE_SHADERS,
                )
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            for &format in CORE_DEPTH_STENCIL_FORMATS {
                for target in [
                    RenderTarget::Array { mip: 0, layer: 0 },
                    RenderTarget::Array { mip: 1, layer: 1 },
                    RenderTarget::Array { mip: 2, layer: 2 },
                ] {
                    check_store_inits_only_target(
                        &ctx,
                        depth_stencil_2d_spec(format),
                        TextureViewDimension::D2,
                        target,
                        false,
                    )
                    .await;
                }
            }
        });

// Rendering to the last mip level of the 3D texture, which has a single depth slice, covers
// its whole tracking unit, so it must behave like any other fully covered render target.
#[apply(gpu_test!)]
static RENDER_PASS_STORE_INITS_ONLY_TARGET_3D_MIP: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // https://github.com/gfx-rs/wgpu/issues/9184
                .expect_fail(
                    FailureCase::molten_vk()
                        .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
                ),
        )
        .run_async(|ctx| async move {
            check_store_inits_only_target(
                &ctx,
                color_3d_spec(),
                TextureViewDimension::D3,
                RenderTarget::Volume {
                    mip: SUBRESOURCE_MIPS - 1,
                    depth_slice: 0,
                },
                false,
            )
            .await;
        });

// `LoadOp::Load` of a single depth slice must initialize the whole mip level, because the
// slices of a 3D mip level are not tracked separately.
#[apply(gpu_test!)]
static RENDER_PASS_LOAD_INITS_WHOLE_3D_MIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // https://github.com/gfx-rs/wgpu/issues/9184
            .expect_fail(
                FailureCase::molten_vk()
                    .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
            ),
    )
    .run_async(|ctx| async move {
        for pre_init in [PreInit::Fresh, PreInit::Sentinel] {
            for target in [
                RenderTarget::Volume {
                    mip: 0,
                    depth_slice: 2,
                },
                RenderTarget::Volume {
                    mip: 1,
                    depth_slice: 1,
                },
            ] {
                // A freshly created texture must read back as zero everywhere. A texture
                // that was fully written must keep its contents everywhere, including the
                // slices that the pass did not target.
                let expected = match pre_init {
                    PreInit::Fresh => Expected::Zero,
                    PreInit::Sentinel => Expected::Sentinel,
                };

                let mut case = RenderTargetInitCase::new(
                    &ctx,
                    color_3d_spec(),
                    TextureViewDimension::D3,
                    pre_init,
                    format!("load of {target} of a 3D texture ({expected:?})"),
                );
                case.create_command_encoder();
                case.pass(target, PassOps::load_store());
                case.submit_command_encoder();

                case.create_command_encoder();
                case.copy_texture_to_buffer();
                case.submit_command_encoder();

                case.assert_contents(|_, _| expected).await;
            }
        }
    });

// Discarding a (mip level, array layer) must reset only that subresource, whether the
// discarded contents are read in the same encoder or after a submit.
#[apply(gpu_test!)]
static DISCARDING_COLOR_TARGET_AT_NONZERO_MIP_AND_LAYER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        // https://github.com/gfx-rs/wgpu/issues/10162
        .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
        .run_async(|ctx| async move {
            for target in [
                RenderTarget::Array { mip: 1, layer: 1 },
                RenderTarget::Array { mip: 2, layer: 2 },
            ] {
                for load in [LoadOp::Load, LoadOp::Clear(())] {
                    for readback_in_same_encoder in [false, true] {
                        check_discard_resets_only_target(
                            &ctx,
                            color_2d_spec(),
                            TextureViewDimension::D2,
                            target,
                            load,
                            readback_in_same_encoder,
                        )
                        .await;
                    }
                }
            }
        });

#[apply(gpu_test!)]
static DISCARDING_DEPTH_TARGET_AT_NONZERO_MIP_AND_LAYER: GpuTestConfiguration =
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
            for &format in CORE_DEPTH_STENCIL_FORMATS {
                check_depth_discard_at_nonzero_mip_and_layer(&ctx, format).await;
            }
        });

// As above, for the depth/stencil format that is behind an optional feature.
#[apply(gpu_test!)]
static DISCARDING_DEPTH_TARGET_AT_NONZERO_MIP_AND_LAYER_DEPTH32FLOAT_STENCIL8:
    GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::DEPTH32FLOAT_STENCIL8)
            .downlevel_flags(DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        check_depth_discard_at_nonzero_mip_and_layer(&ctx, TextureFormat::Depth32FloatStencil8)
            .await;
    });

// Init state is not tracked per aspect, so discarding one aspect of a combined
// depth/stencil texture must still leave the discarded aspect of that subresource
// initialized, and must not affect any other subresource.
#[apply(gpu_test!)]
static DISCARDING_EITHER_DEPTH_OR_STENCIL_ASPECT_AT_NONZERO_MIP_AND_LAYER: GpuTestConfiguration =
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
            let target = RenderTarget::Array { mip: 1, layer: 1 };
            let mut case = RenderTargetInitCase::new(
                &ctx,
                depth_stencil_2d_spec(TextureFormat::Depth24PlusStencil8),
                TextureViewDimension::D2,
                PreInit::Sentinel,
                format!("diverging depth/stencil discard of {target}"),
            );

            case.create_command_encoder();
            case.pass(target, PassOps::discard_depth_keep_stencil());
            case.submit_command_encoder();

            case.create_command_encoder();
            case.pass(target, PassOps::discard_stencil_keep_depth());
            case.submit_command_encoder();

            case.create_command_encoder();
            case.copy_texture_to_buffer();
            case.submit_command_encoder();

            case.assert_only_target(target, Expected::Zero).await;
        });

// Discarding both aspects of a combined depth/stencil texture must reset the whole
// subresource, even when the load operations of the two aspects diverge. This is a
// regression test for an error in the handling of divergent depth/stencil usage within a
// render pass, which takes a dedicated path for handling the store ops, even when they do
// not differ.
#[apply(gpu_test!)]
static DISCARDING_BOTH_DEPTH_AND_STENCIL_WITH_DIVERGING_LOAD_OPS: GpuTestConfiguration =
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
            let target = RenderTarget::Array { mip: 1, layer: 1 };
            let mut case = RenderTargetInitCase::new(
                &ctx,
                depth_stencil_2d_spec(TextureFormat::Depth24PlusStencil8),
                TextureViewDimension::D2,
                PreInit::Sentinel,
                format!("discard of both aspects of {target}, loading stencil"),
            );

            case.create_command_encoder();
            case.pass(
                target,
                PassOps::discard_both_with_diverging_load(/* load_depth */ false),
            );
            case.submit_command_encoder();

            case.create_command_encoder();
            case.copy_texture_to_buffer();
            case.submit_command_encoder();

            case.assert_only_target(target, Expected::Zero).await;
        });

// A discarded subresource that is loaded by a later render pass in the same encoder must be
// initialized before that pass runs.
#[apply(gpu_test!)]
static DISCARDING_COLOR_TARGET_THEN_LOADING_IT_IN_SAME_ENCODER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        // https://github.com/gfx-rs/wgpu/issues/10162
        .parameters(TestParameters::default().expect_fail(FailureCase::webgl2()))
        .run_async(|ctx| async move {
            let target = RenderTarget::Array { mip: 1, layer: 1 };
            let mut case = RenderTargetInitCase::new(
                &ctx,
                color_2d_spec(),
                TextureViewDimension::D2,
                PreInit::Sentinel,
                format!("discard then load of {target}"),
            );

            case.create_command_encoder();
            case.pass(target, PassOps::load_discard());
            case.pass(target, PassOps::load_store());
            case.copy_texture_to_buffer();
            case.submit_command_encoder();

            case.assert_only_target(target, Expected::Zero).await;
        });

// The slices of a 3D mip level are not tracked separately, so a `StoreOp::Store` pass that
// covers a single slice of a multi-slice mip level must trigger initialization of the
// complete texture uninitialized.
#[apply(gpu_test!)]
static RENDER_PASS_STORE_TO_3D_DEPTH_SLICE_INITS_OTHER_SLICES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // https://github.com/gfx-rs/wgpu/issues/9184
                .expect_fail(
                    FailureCase::molten_vk()
                        .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
                ),
        )
        .run_async(|ctx| async move {
            check_store_inits_only_target(
                &ctx,
                color_3d_spec(),
                TextureViewDimension::D3,
                RenderTarget::Volume {
                    mip: 0,
                    depth_slice: 2,
                },
                false,
            )
            .await;
        });

// Discarding a single slice of a multi-slice 3D mip level must reset only that slice. Since
// the init tracker cannot represent a discarded slice, the slice has to be reinitialized
// immediately instead.
#[apply(gpu_test!)]
static DISCARDING_3D_DEPTH_SLICE_PRESERVES_OTHER_SLICES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // https://github.com/gfx-rs/wgpu/issues/10162
                .expect_fail(FailureCase::webgl2())
                // https://github.com/gfx-rs/wgpu/issues/9184
                .expect_fail(
                    FailureCase::molten_vk()
                        .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
                ),
        )
        .run_async(|ctx| async move {
            for load in [LoadOp::Load, LoadOp::Clear(())] {
                for readback_in_same_encoder in [false, true] {
                    check_discard_resets_only_target(
                        &ctx,
                        color_3d_spec(),
                        TextureViewDimension::D3,
                        RenderTarget::Volume {
                            mip: 0,
                            depth_slice: 2,
                        },
                        load,
                        readback_in_same_encoder,
                    )
                    .await;
                }
            }
        });

// Distinct depth slices of one 3D mip level do not overlap, so a single render pass may
// attach several of them at once. Those attachments are processed by successive calls to
// [`CommandBufferTextureMemoryActions::register_init_action`]. Global init tracking for 3D
// textures operates per mip level, but pending discards within a single command buffer
// _are_ tracked per depth slice, and it is important that they be matched that way against
// init actions, due to the following case.
//
// A pending discard of a slice is repaired by clearing it ahead of a subsequent operation in
// the same command buffer that needs the mip level initialized, or otherwise at the end of
// the command buffer. Doing that without checking depth slices is correct when the discard
// came from an earlier pass, but not when it came from another attachment to the same
// pass:
//
// - attachment 0: mip 0, depth_slice 0, `LoadOp::Clear` / `StoreOp::Discard`
// - attachment 1: mip 0, depth_slice 1, `LoadOp::Load` / `StoreOp::Store`
//
// Attachment 1 needs mip 0 initialized, so if depth slices were not considered, it would
// repair slice 0's discard. But that clear would be emitted ahead of the pass, so slice 0
// would be left holding the discarded contents with the mip level recorded as initialized.
// The repair has to be deferred to the end of the command buffer instead. We test both
// `Load` and `Clear` for attachment 1 (`other`). Either requires that the whole mip level
// be initialized, because a mip level is the finest granularity the init tracker can record
// (after the command buffer concludes).
//
// Attachment 0 clears to the sentinel value so that slice 0 holds something other than zero
// when the discard takes effect. Under `LoadOp::Load` the pass would write nothing there,
// and a misplaced pre-pass clear to zero would coincidentally match the expected result,
// leaving the test unable to tell whether the discard was correctly tracked.
#[apply(gpu_test!)]
static DISCARDING_3D_DEPTH_SLICE_ALONGSIDE_ANOTHER_SLICE_IN_SAME_PASS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // https://github.com/gfx-rs/wgpu/issues/10162
                .expect_fail(FailureCase::webgl2())
                // https://github.com/gfx-rs/wgpu/issues/9184
                .expect_fail(
                    FailureCase::molten_vk()
                        .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
                ),
        )
        .run_async(|ctx| async move {
            let discarded = RenderTarget::Volume {
                mip: 0,
                depth_slice: 0,
            };
            let other = RenderTarget::Volume {
                mip: 0,
                depth_slice: 1,
            };

            for (other_ops, other_desc) in [
                (
                    PassOps::clear_store(sentinel_for(color_3d_spec().format)),
                    "clear + store",
                ),
                (PassOps::load_store(), "load + store"),
            ] {
                let mut case = RenderTargetInitCase::new(
                    &ctx,
                    color_3d_spec(),
                    TextureViewDimension::D3,
                    PreInit::Sentinel,
                    format!("clear + discard of {discarded} alongside {other_desc} of {other}"),
                );

                let sentinel = case.sentinel;
                case.create_command_encoder();
                case.multi_slice_pass(&[
                    (discarded, PassOps::clear_discard(sentinel)),
                    (other, other_ops),
                ]);
                case.submit_command_encoder();

                case.create_command_encoder();
                case.copy_texture_to_buffer();
                case.submit_command_encoder();

                // Every slice the pass did not discard holds the sentinel: the ones it did not
                // touch keep what `PreInit::Sentinel` wrote, and the other attachment either
                // cleared its slice to the sentinel or loaded and stored it unchanged.
                case.assert_only_target(discarded, Expected::Zero).await;
            }
        });

// Tests that a full-extent, single-aspect `write_texture` does not cause
// *other* aspects of a multi-aspect texture to be considered initialized.

#[apply(gpu_test!)]
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
#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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
#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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
#[apply(gpu_test!)]
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
#[apply(gpu_test!)]
static COPY_TEXTURE_TO_BUFFER_3D_SOURCE_ORIGIN_Z_UNINIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_source_init(&ctx, ReadMethod::CopyTextureToBuffer).await;
        });

#[apply(gpu_test!)]
static COPY_TEXTURE_TO_TEXTURE_3D_SOURCE_ORIGIN_Z_UNINIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_source_init(&ctx, ReadMethod::CopyTextureToTexture).await;
        });

// The first depth slice must be initialized to zero before a partial copy into a fresh 3D
// texture with destination `origin.z >= 1`.
#[apply(gpu_test!)]
static COPY_BUFFER_TO_TEXTURE_3D_DEST_ORIGIN_Z_PARTIAL: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            check_3d_copy_dest_init(&ctx, WriteMethod::CopyBufferToTexture).await;
        });

#[apply(gpu_test!)]
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

// Tests that the padding (tail) at the end of a buffer allocation is
// zero-initialized.
//
// The test is effective mainly on Vulkan, where it can exploit the fact that
// `vkCmdBindVertexBuffers` does not specify the end of the binding, so a vertex
// shader can fetch a vertex whose bytes lie in the tail. Draw-time bounds
// checking does not limit the vertex count for indexed draws, so we use an
// indexed draw whose largest index points at the buffer tail.

const VB_TAIL_VERTEX_STRIDE: u64 = 4;
const VB_TAIL_VISIBLE_VERTS: u32 = 4;

const VB_TAIL_SHADER: &str = "
    @group(0) @binding(0) var<storage, read_write> result: array<u32>;

    @vertex
    fn vs_main(@builtin(vertex_index) ix: u32, @location(0) value: u32) -> @builtin(position) vec4f {
        result[ix] = value;
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
";

// `MAP_WRITE` + `VERTEX` requires `MAPPABLE_PRIMARY_BUFFERS`.
#[apply(gpu_test!)]
static VERTEX_BUFFER_TAIL_INIT_PLAIN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::VERTEX_WRITABLE_STORAGE)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        for app_writes in [false, true] {
            check_vertex_buffer_tail_init(&ctx, false, false, app_writes).await;
        }
    });

#[apply(gpu_test!)]
static VERTEX_BUFFER_TAIL_INIT_MAP_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::VERTEX_WRITABLE_STORAGE | Features::MAPPABLE_PRIMARY_BUFFERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        for app_writes in [false, true] {
            check_vertex_buffer_tail_init(&ctx, true, false, app_writes).await;
        }
    });

#[apply(gpu_test!)]
static VERTEX_BUFFER_TAIL_INIT_MAPPED_AT_CREATION: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::VERTEX_WRITABLE_STORAGE)
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            for app_writes in [false, true] {
                check_vertex_buffer_tail_init(&ctx, false, true, app_writes).await;
            }
        });

#[apply(gpu_test!)]
static VERTEX_BUFFER_TAIL_INIT_MAP_WRITE_MAPPED_AT_CREATION: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::VERTEX_WRITABLE_STORAGE | Features::MAPPABLE_PRIMARY_BUFFERS)
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            for app_writes in [false, true] {
                check_vertex_buffer_tail_init(&ctx, true, true, app_writes).await;
            }
        });

async fn check_vertex_buffer_tail_init(
    ctx: &TestingContext,
    map_write: bool,
    mapped_at_creation: bool,
    app_writes: bool,
) {
    let case_desc = format!(
        "map_write={map_write}, mapped_at_creation={mapped_at_creation}, app_writes={app_writes}"
    );

    let vb_size = VB_TAIL_VISIBLE_VERTS as u64 * VB_TAIL_VERTEX_STRIDE;
    let vertex_data: Vec<u32> = (0..VB_TAIL_VISIBLE_VERTS)
        .map(|i| 0xA0A0_0000 + i)
        .collect();
    let vertex_data_bytes: &[u8] = bytemuck::cast_slice(&vertex_data);

    // Add `COPY_DST` if the application needs it to write via the queue;
    // it does not affect which tail-init path is taken.
    let mut usage = BufferUsages::VERTEX;
    if map_write {
        usage |= BufferUsages::MAP_WRITE;
    } else if app_writes && !mapped_at_creation {
        usage |= BufferUsages::COPY_DST;
    }

    let vertex_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("vertex buffer under test"),
        size: vb_size,
        usage,
        mapped_at_creation,
    });

    // Fill (or leave zero) the application-visible portion of the buffer.
    if mapped_at_creation {
        // The buffer is mapped at creation; write into it if desired, then unmap.
        if app_writes {
            let mut view = vertex_buffer.slice(..).get_mapped_range_mut().unwrap();
            view.copy_from_slice(vertex_data_bytes);
        }
        vertex_buffer.unmap();
    } else if app_writes {
        if map_write {
            vertex_buffer.slice(..).map_async(MapMode::Write, |_| ());
            ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
            {
                let mut view = vertex_buffer.slice(..).get_mapped_range_mut().unwrap();
                view.copy_from_slice(vertex_data_bytes);
            }
            vertex_buffer.unmap();
        } else {
            ctx.queue.write_buffer(&vertex_buffer, 0, vertex_data_bytes);
        }
    }

    // Index buffer whose largest index (`VB_TAIL_VISIBLE_VERTS`) points one vertex
    // past the visible area, into the tail.
    let indices: Vec<u32> = (0..=VB_TAIL_VISIBLE_VERTS).collect();
    let index_buffer = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    // The vertex shader writes each fetched value into this buffer at `vertex_index`.
    let result_len = indices.len();
    let result_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("result buffer"),
        size: (result_len * core::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("result readback"),
        size: result_buffer.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let output_texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("unused render attachment"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&Default::default());

    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vertex buffer tail shader"),
        source: ShaderSource::Wgsl(VB_TAIL_SHADER.into()),
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("vertex buffer tail pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: VB_TAIL_VERTEX_STRIDE,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[VertexAttribute {
                        format: VertexFormat::Uint32,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(output_texture.format().into())],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: result_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("vertex buffer tail pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::default()),
                    store: StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..result_len as u32, 0, 0..1);
    }
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback, 0, result_buffer.size());
    ctx.queue.submit([encoder.finish()]);

    let data = map_and_read(ctx, &readback).await;
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

    // Visible vertices should match vertex data if the application wrote it, or
    // be zero if not.
    for i in 0..VB_TAIL_VISIBLE_VERTS as usize {
        let got = result[i];
        if app_writes {
            assert_eq!(
                got, vertex_data[i],
                "did not read expected non-zero data in visible area \
                 (case: {case_desc}, vertex {i}): expected 0x{:08x}, got 0x{got:08x}",
                vertex_data[i],
            );
        } else {
            assert_eq!(
                got, 0,
                "did not read expected zero (uninitialized) data in visible area \
                 (case: {case_desc}, vertex {i}): got 0x{got:08x}",
            );
        }
    }

    // Tail vertex: never written by the application, must be zero-initialized.
    let tail = result[VB_TAIL_VISIBLE_VERTS as usize];
    assert_eq!(
        tail, 0,
        "did not read expected zero data in padding area (case: {case_desc}): got 0x{tail:08x}",
    );
}

// Tests that the padding of a `copy_texture_to_buffer()` destination is not filled
// with the leftovers of previously freed memory.
//
// The test is effective mainly on DX12, where a destination offset that is not a
// multiple of `D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT` (512) forces the copy to go
// through a backend-private intermediate buffer. The texture copy only writes the
// texel bytes of each row of that buffer, but the whole buffer is then copied into
// the destination, so the padding of the intermediate buffer has to be zeroed first.

// A single pixel wide texture, so that each row of the copy is 4 bytes of texel data
// followed by 252 bytes of padding.
#[apply(gpu_test!)]
static COPY_TEXTURE_TO_BUFFER_UNALIGNED_OFFSET_ROW_PADDING_INIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            test_copy_texture_to_buffer_padding_init(
                &ctx,
                TextureFormat::Rgba8Unorm,
                Extent3d {
                    width: 1,
                    height: 512,
                    depth_or_array_layers: 1,
                },
                256,
                512,
            )
            .await;
        });

// Padding that neither starts nor ends at a four byte boundary, plus padding
// between the array layers.
#[apply(gpu_test!)]
static COPY_TEXTURE_TO_BUFFER_UNALIGNED_OFFSET_IMAGE_PADDING_INIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().limits(Limits::downlevel_defaults()))
        .run_async(|ctx| async move {
            test_copy_texture_to_buffer_padding_init(
                &ctx,
                TextureFormat::R8Unorm,
                Extent3d {
                    width: 3,
                    height: 8,
                    depth_or_array_layers: 3,
                },
                256,
                12,
            )
            .await;
        });

async fn test_copy_texture_to_buffer_padding_init(
    ctx: &TestingContext,
    format: TextureFormat,
    size: Extent3d,
    bytes_per_row: u32,
    rows_per_image: u32,
) {
    /// A legal `copy_texture_to_buffer()` destination offset that is not a multiple of
    /// D3D12's 512 byte texture data placement alignment.
    const T2B_PAD_OFFSET: u64 = 4;
    /// How many copies to perform, to give the allocator several chances to hand out
    /// the memory of the freed seed buffers.
    const T2B_PAD_COPIES: u64 = 8;

    let texel_bytes = format.block_copy_size(None).unwrap();
    let row_bytes = size.width * texel_bytes;
    let image_stride = u64::from(bytes_per_row) * u64::from(rows_per_image);
    let image_bytes = u64::from(bytes_per_row) * u64::from(size.height - 1) + u64::from(row_bytes);
    // The copy footprint does not include the padding after its very last row.
    let footprint = u64::from(size.depth_or_array_layers - 1) * image_stride + image_bytes;
    // Round up so that every copy's destination offset is congruent to
    // `T2B_PAD_OFFSET` modulo 512, and thus unaligned for DX12.
    let stride = footprint.next_multiple_of(512);

    // Dirty some device memory with a recognizable pattern and release it again, so
    // that the buffers the backend allocates for the copies below are likely to be
    // suballocated on top of it.
    let markers: Vec<u32> = (0..footprint.next_multiple_of(4) as usize / 4)
        .map(|word_index| 0xA73C_0001 + word_index as u32)
        .collect();
    let seed_buffers: Vec<Buffer> = (0..T2B_PAD_COPIES)
        .map(|i| {
            ctx.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some(&format!("copy padding seed {i}")),
                contents: bytemuck::cast_slice(&markers),
                usage: BufferUsages::COPY_SRC,
            })
        })
        .collect();
    ctx.queue.submit(None);
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    for seed_buffer in &seed_buffers {
        seed_buffer.destroy();
    }
    drop(seed_buffers);
    ctx.queue.submit(None);
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("copy padding source"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("copy padding readback"),
        size: T2B_PAD_OFFSET + stride * T2B_PAD_COPIES,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    for i in 0..T2B_PAD_COPIES {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: T2B_PAD_OFFSET + stride * i,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows_per_image),
                },
            },
            size,
        );
    }
    ctx.queue.submit([encoder.finish()]);

    // Neither the source texture nor the destination buffer was ever written by the
    // application, so the whole buffer must read back as zero. The interesting part is
    // the footprint of each copy, texel data and padding alike: the copies mark it
    // initialized, so mapping the buffer will not zero it for us.
    let data = map_and_read(ctx, &readback).await;
    assert!(
        data.iter().all(|&byte| byte == 0),
        "the destination buffer of copies from a never-written texture is not all zero",
    );
}

#[apply(gpu_test!)]
static MARK_EXTERNALLY_INITIALIZED_SKIPS_LAZY_CLEAR: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(Limits::downlevel_defaults())
                // This test pokes at raw hal texture state, which is only
                // exercised below for the desktop-native backends.
                .skip(FailureCase::backend(
                    Backends::all() - (Backends::VULKAN | Backends::METAL | Backends::DX12),
                )),
        )
        .run_async(|ctx| async move {
            match ctx.adapter_info.backend {
                #[cfg(any(
                    target_os = "windows",
                    target_os = "linux",
                    target_os = "android",
                    target_os = "freebsd"
                ))]
                Backend::Vulkan => {
                    check_mark_externally_initialized::<hal::vulkan::Api>(&ctx).await;
                }
                #[cfg(target_vendor = "apple")]
                Backend::Metal => {
                    check_mark_externally_initialized::<hal::metal::Api>(&ctx).await;
                }
                #[cfg(target_os = "windows")]
                Backend::Dx12 => {
                    check_mark_externally_initialized::<hal::dx12::Api>(&ctx).await;
                }
                other => unreachable!(
                    "test is configured to skip all backends except Vulkan/Metal/Dx12, \
                     but ran on {other:?}"
                ),
            }
        });

#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_vendor = "apple"
))]
async fn check_mark_externally_initialized<A: hal::Api>(ctx: &TestingContext) {
    check_mark_externally_initialized_case::<A>(ctx, false).await;
    check_mark_externally_initialized_case::<A>(ctx, true).await;
}

#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_vendor = "apple"
))]
async fn check_mark_externally_initialized_case<A: hal::Api>(ctx: &TestingContext, mark: bool) {
    use core::iter;
    use wgpu::hal::{CommandEncoder as _, Device as _};

    const SENTINEL: u8 = 0xAB;
    // `Rgba8Unorm` at this width gives exactly `COPY_BYTES_PER_ROW_ALIGNMENT` bytes per
    // row, so the raw hal copy below needs no extra row padding.
    let size = Extent3d {
        width: COPY_BYTES_PER_ROW_ALIGNMENT / 4,
        height: 4,
        depth_or_array_layers: 1,
    };
    let bytes_per_row = size.width * 4;
    let buffer_size = u64::from(bytes_per_row * size.height);

    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("mark_externally_initialized target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Write a non-zero pattern into the texture through its raw hal handle, entirely
    // bypassing wgpu-core's command recording (and thus its init tracking) for the
    // texture. This simulates e.g. a video decoder writing into the texture through
    // native driver APIs.
    //
    // SAFETY: `hal_device`, `hal_texture`, and `raw_encoder` are all obtained from the
    // same wgpu `Device`, and are only used to write and destroy resources not tracked
    // by wgpu-core.
    unsafe {
        let hal_device = ctx.device.as_hal::<A>().expect("adapter backend mismatch");
        let hal_texture = texture.as_hal::<A>().expect("adapter backend mismatch");

        let staging_buffer = hal_device
            .create_buffer(&hal::BufferDescriptor {
                label: Some("mark_externally_initialized staging"),
                size: buffer_size,
                usage: wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
                memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
            })
            .expect("failed to create staging buffer");
        {
            let mapping = hal_device
                .map_buffer(&staging_buffer, 0..buffer_size)
                .expect("failed to map staging buffer");
            core::ptr::write_bytes(mapping.ptr.as_ptr(), SENTINEL, buffer_size as usize);
            if !mapping.is_coherent {
                hal_device.flush_mapped_ranges(&staging_buffer, iter::once(0..buffer_size));
            }
            hal_device.unmap_buffer(&staging_buffer);
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.as_hal_mut::<A, _, ()>(|raw_encoder| {
            let raw_encoder = raw_encoder.expect("adapter backend mismatch");
            raw_encoder.transition_textures(iter::once(hal::TextureBarrier {
                texture: &*hal_texture,
                range: wgt::ImageSubresourceRange::default(),
                usage: hal::StateTransition {
                    from: wgt::TextureUses::UNINITIALIZED,
                    to: wgt::TextureUses::COPY_DST,
                },
                queue_family_ownership_transfer: None,
            }));
            raw_encoder.copy_buffer_to_texture(
                &staging_buffer,
                &*hal_texture,
                iter::once(hal::BufferTextureCopy {
                    buffer_layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: None,
                    },
                    texture_base: hal::TextureCopyBase {
                        mip_level: 0,
                        array_layer: 0,
                        origin: Origin3d::ZERO,
                        aspect: hal::FormatAspects::COLOR,
                    },
                    size: hal::CopyExtent {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    },
                }),
            );
            raw_encoder.transition_textures(iter::once(hal::TextureBarrier {
                texture: &*hal_texture,
                range: wgt::ImageSubresourceRange::default(),
                usage: hal::StateTransition {
                    from: wgt::TextureUses::COPY_DST,
                    to: wgt::TextureUses::COPY_SRC,
                },
                queue_family_ownership_transfer: None,
            }));
        });
        // Release the texture's snatch-lock guard before submitting: queue submission
        // needs to acquire it too, and it is not reentrant.
        drop(hal_texture);
        ctx.queue.submit(Some(encoder.finish()));
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

        hal_device.destroy_buffer(staging_buffer);
    }

    if mark {
        unsafe { texture.mark_externally_initialized() };
    }

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &texture);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    readback_buffers.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit(Some(encoder.finish()));

    if mark {
        assert!(
            !readback_buffers.are_zero(ctx).await,
            "texture written through as_hal and marked externally initialized was \
             still lazily cleared to zero before being read back",
        );
    } else {
        assert!(
            readback_buffers.are_zero(ctx).await,
            "texture written through as_hal without being marked externally initialized \
             was not lazily cleared to zero as expected",
        );
    }
}
