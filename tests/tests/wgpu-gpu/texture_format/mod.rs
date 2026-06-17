use exhaust::Exhaust;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compat_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compat_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compat_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_native_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_native_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_native_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_norm_16bit_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_norm_16bit_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_norm_16bit_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_bc_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_bc_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_bc_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_etc2_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_etc2_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_etc2_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_hdr_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_hdr_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_hdr_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_nv12_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_nv12_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_nv12_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_p010_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_p010_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_p010_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_rg11b10ufloat_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_rg11b10ufloat_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_rg11b10ufloat_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_depth32floatstencil8_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_depth32floatstencil8_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_depth32floatstencil8_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_r64uint_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_r64uint_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_r64uint_multisample_2d_array);

    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_bgra8unorm_atomic);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_bgra8unorm_no_features);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_bgra8unorm_multisample_2d_array);

    // Extra tests
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_astc_compression_astc_3d);
    tests.push(TEXTURE_FORMAT_CAPABILITIES_CHECK_compression_bc_compression_bc_3d);
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum UsageType {
    Standard,
    Multisample,
    Transient,
    D3,
    Array,
    MultisampleArray,
}

const NORMAL_USAGE_TYPES: &[UsageType] = &[
    UsageType::Standard,
    UsageType::Multisample,
    UsageType::Transient,
    UsageType::D3,
    UsageType::Array,
];

fn texture_format_capabilities_check(
    ctx: TestingContext,
    iter: impl Iterator<Item = wgpu::TextureFormat>,
    format_features: wgpu::Features,
    caps_features: wgpu::Features,
    usage_types: &[UsageType],
) {
    use wgpu::TextureFormatFeatureFlags as FF;
    use wgpu::TextureUsages as TU;

    let empty_features = caps_features.is_empty();
    let all_multisample_flags = FF::MULTISAMPLE_X16
        | FF::MULTISAMPLE_X8
        | FF::MULTISAMPLE_X4
        | FF::MULTISAMPLE_X2
        | FF::MULTISAMPLE_RESOLVE;

    const TEXTURE_DIM: u32 = 16;

    let mut any_error = false;
    let mut errors = Vec::new();

    for format in iter {
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let (actual_caps, caps) = get_caps(&ctx, format, &mut errors);
        for &usage_type in usage_types {
            let mut caps = caps;

            if !format_features.contains(format.required_features()) {
                panic!(
                    "Format {format:?} features not respected in test: requires {} but supplied {}",
                    format.required_features(),
                    format_features
                );
            }
            let can_3d = if format.has_depth_aspect() || format.has_stencil_aspect() {
                false
            } else if format.is_compressed() {
                if format == wgpu::TextureFormat::Rgb9e5Ufloat {
                    true
                } else if format.is_astc() {
                    ctx.device_features
                        .contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
                } else if format.is_bcn() {
                    ctx.device_features
                        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
                } else {
                    false
                }
            } else {
                false
            };

            let mut depth = 1;
            let mut texture_dimension = wgpu::TextureDimension::D2;
            let mut view_dim = wgpu::TextureViewDimension::D2;
            let mut array_layer_count = None;
            match usage_type {
                UsageType::Multisample => {
                    caps.allowed_usages
                        .remove(TU::STORAGE_BINDING | TU::STORAGE_ATOMIC);
                    if !caps.flags.intersects(all_multisample_flags) {
                        continue;
                    }
                }
                UsageType::MultisampleArray => {
                    caps.allowed_usages
                        .remove(TU::STORAGE_BINDING | TU::STORAGE_ATOMIC);
                    depth = 4;
                    view_dim = wgpu::TextureViewDimension::D2Array;
                    array_layer_count = Some(4);
                    if !can_3d
                        || !caps.flags.intersects(all_multisample_flags)
                        || !ctx
                            .device_features
                            .contains(wgpu::Features::MULTISAMPLE_ARRAY)
                    {
                        continue;
                    }
                }
                UsageType::Transient => {
                    caps.allowed_usages &= TU::TRANSIENT | TU::RENDER_ATTACHMENT;
                    if caps.allowed_usages.is_empty() {
                        continue;
                    }
                }
                UsageType::Array => {
                    depth = 4;
                    view_dim = wgpu::TextureViewDimension::D2Array;
                    array_layer_count = Some(4);
                }
                UsageType::D3 => {
                    if !can_3d {
                        continue;
                    }
                    depth = 4;
                    texture_dimension = wgpu::TextureDimension::D3;
                    view_dim = wgpu::TextureViewDimension::D3;
                }
                UsageType::Standard => (),
            }
            if usage_type != UsageType::Multisample && usage_type != UsageType::MultisampleArray {
                caps.flags.remove(all_multisample_flags);
            }
            if usage_type != UsageType::Transient {
                caps.allowed_usages.remove(TU::TRANSIENT);
            }

            //let supports_transient = caps.allowed_usages.contains(wgpu::TextureUsages::TRANSIENT);

            let sample_count = if caps.flags.contains(FF::MULTISAMPLE_X16) {
                16
            } else if caps.flags.contains(FF::MULTISAMPLE_X8) {
                8
            } else if caps.flags.contains(FF::MULTISAMPLE_X4) {
                4
            } else if caps.flags.contains(FF::MULTISAMPLE_X2) {
                2
            } else {
                1
            };

            // See <https://github.com/gfx-rs/wgpu/issues/9344>
            // But basically, some formats you have to look at one specific aspect for a view
            // which has its own texture format.
            let mut desc_view_formats = Vec::new();
            let mut view_formats_aspects = Vec::new();
            if format
                .aspect_specific_format(wgpu::TextureAspect::All)
                .is_none()
            {
                for aspect in [
                    wgpu::TextureAspect::Plane0,
                    wgpu::TextureAspect::Plane1,
                    wgpu::TextureAspect::Plane2,
                    wgpu::TextureAspect::DepthOnly,
                    wgpu::TextureAspect::StencilOnly,
                ] {
                    if let Some(inner) = format.aspect_specific_format(aspect) {
                        view_formats_aspects.push((inner, aspect));
                    }
                }
            } else if ctx
                .adapter_downlevel_capabilities
                .flags
                .contains(wgpu::DownlevelFlags::VIEW_FORMATS)
                && usage_type != UsageType::Transient
            {
                for other_format in wgpu::TextureFormat::exhaust() {
                    if other_format.remove_srgb_suffix() == format.remove_srgb_suffix() {
                        view_formats_aspects.push((other_format, wgpu::TextureAspect::All));
                        if other_format != format {
                            desc_view_formats.push(other_format);
                        }
                    }
                }
            } else {
                view_formats_aspects.push((format, wgpu::TextureAspect::All));
            }

            eprintln!("Creating texture with format {format:?} and caps {actual_caps:?}");
            let texture_desc = wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: TEXTURE_DIM,
                    height: TEXTURE_DIM,
                    depth_or_array_layers: depth,
                },
                mip_level_count: 1,
                sample_count,
                dimension: texture_dimension,
                format,
                usage: caps.allowed_usages,
                view_formats: &desc_view_formats,
            };
            let texture1 = ctx.device.create_texture(&texture_desc);

            // We only want each test to run once. So atomic tests won't run in both non-featured and fully-featured tests
            if empty_features {
                for &(other_format, aspect) in &view_formats_aspects {
                    let (other_actual_caps, other_caps) =
                        get_caps(&ctx, other_format, &mut Vec::new());
                    eprintln!("Trying to create view with format {other_format:?} on texture with format {format:?} and caps {actual_caps:?}, and view format caps {other_actual_caps:?}");
                    let _view = texture1.create_view(&wgpu::TextureViewDescriptor {
                        label: None,
                        format: Some(other_format),
                        dimension: Some(view_dim),
                        usage: Some(caps.allowed_usages & other_caps.allowed_usages),
                        aspect,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: 0,
                        array_layer_count,
                    });
                }

                if caps.allowed_usages.contains(TU::COPY_SRC | TU::COPY_DST) {
                    eprintln!("Encoding copy");
                    let texture2 = ctx.device.create_texture(&texture_desc);
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfoBase {
                            texture: &texture1,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfoBase {
                            texture: &texture2,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: TEXTURE_DIM,
                            height: TEXTURE_DIM,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                if caps.allowed_usages.contains(TU::RENDER_ATTACHMENT) {
                    eprintln!("Rendering into it");
                    smoke_render_clear(&mut encoder, &texture1, format);
                }
                if caps.allowed_usages.contains(TU::STORAGE_BINDING) {
                    eprintln!("Binding it as a storage texture");
                    let access = if caps.flags.contains(FF::STORAGE_WRITE_ONLY) {
                        wgpu::StorageTextureAccess::WriteOnly
                    } else if caps.flags.contains(FF::STORAGE_READ_WRITE) {
                        wgpu::StorageTextureAccess::ReadWrite
                    } else {
                        wgpu::StorageTextureAccess::ReadOnly
                    };
                    smoke_storage_bind(
                        &ctx,
                        &texture1,
                        format,
                        view_dim,
                        array_layer_count,
                        access,
                    );
                }
                if caps.allowed_usages.contains(TU::STORAGE_ATOMIC) {
                    eprintln!("Binding it as an atomic storage texture");
                    smoke_storage_bind(
                        &ctx,
                        &texture1,
                        format,
                        view_dim,
                        array_layer_count,
                        wgpu::StorageTextureAccess::Atomic,
                    );
                }
                if caps.flags.intersects(
                    FF::MULTISAMPLE_X2 | FF::MULTISAMPLE_X4 | FF::MULTISAMPLE_X8 | FF::MULTISAMPLE_X16,
                ) {
                    // `texture1` already has the chosen `sample_count`; one clear
                    // exercises the multisampled path.
                    eprintln!("Rendering into it multisampled");
                    smoke_render_clear(&mut encoder, &texture1, format);
                }
                if caps.flags.contains(FF::MULTISAMPLE_RESOLVE)
                    && !format.has_depth_aspect()
                    && !format.has_stencil_aspect()
                {
                    eprintln!("Resolving it");
                    smoke_multisample_resolve(&ctx, &mut encoder, &texture1, format, TEXTURE_DIM);
                }
            } else if (ctx
                .device_features
                .contains(wgpu::Features::TEXTURE_INT64_ATOMIC)
                && format == wgpu::TextureFormat::R64Uint)
                || (ctx.device_features.contains(wgpu::Features::TEXTURE_ATOMIC)
                    && [wgpu::TextureFormat::R32Uint, wgpu::TextureFormat::R32Sint]
                        .contains(&format))
            {
                // Multisample usage types strip storage usages above.
                if caps.allowed_usages.contains(TU::STORAGE_ATOMIC) {
                    eprintln!("Binding it as an atomic storage texture");
                    smoke_storage_bind(
                        &ctx,
                        &texture1,
                        format,
                        view_dim,
                        array_layer_count,
                        wgpu::StorageTextureAccess::Atomic,
                    );
                }
            } else if ctx
                .device_features
                .contains(wgpu::Features::MULTISAMPLE_ARRAY)
                && caps.flags.intersects(
                    FF::MULTISAMPLE_X2 | FF::MULTISAMPLE_X4 | FF::MULTISAMPLE_X8 | FF::MULTISAMPLE_X16,
                )
            {
                // `texture1` is a multisampled 2D array here; clear layer 0.
                eprintln!("Rendering into it multisampled in an array");
                smoke_render_clear(&mut encoder, &texture1, format);
            }
        }
        // Submit and then poll immediately. This lets the textures be dropped to avoid OOMs with combinatorial explosion
        eprintln!("Attempting to record & submit command buffer");
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        if !errors.is_empty() && empty_features {
            any_error = true;
            eprintln!("Nonsensical capabilities: {errors:#?}");
            errors.clear();
        }
    }
    if any_error {
        panic!("Nonsensical capabilities above");
    }
}

/// Return the reported caps along with the actually sensible supported caps
/// (removing nonsense capabilities with error messages).
fn get_caps(
    ctx: &TestingContext,
    format: wgpu::TextureFormat,
    errors: &mut Vec<String>,
) -> (wgpu::TextureFormatFeatures, wgpu::TextureFormatFeatures) {
    use wgpu::TextureFormatFeatureFlags as FF;
    use wgpu::TextureUsages as TU;

    let all_multisample_flags = FF::MULTISAMPLE_X16
        | FF::MULTISAMPLE_X8
        | FF::MULTISAMPLE_X4
        | FF::MULTISAMPLE_X2
        | FF::MULTISAMPLE_RESOLVE;
    let all_storage_flags = FF::STORAGE_READ_ONLY | FF::STORAGE_READ_WRITE | FF::STORAGE_WRITE_ONLY;

    // Mirror `wgpu_core`'s `Device::describe_format_features`: the runtime uses
    // the adapter's (backend) format features whenever we either opted into
    // `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` or are running on a downlevel
    // adapter that doesn't guarantee WebGPU texture format support (e.g. GLES).
    // Computing caps any other way makes the test claim usages that the runtime
    // then rejects (see the R32Float `RENDER_ATTACHMENT` downlevel mismatch).
    let using_device_features = ctx
        .device_features
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
    let downlevel = !ctx
        .adapter_downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT);
    let actual_caps = if using_device_features || downlevel {
        ctx.adapter.get_texture_format_features(format)
    } else {
        format.guaranteed_format_features(ctx.device_features)
    };
    let mut caps = actual_caps;
    if format == wgpu::TextureFormat::Rg11b10Ufloat {
        if ctx
            .device_features
            .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
        {
            caps.allowed_usages |= TU::RENDER_ATTACHMENT;
        } else {
            caps.flags
                .remove(all_multisample_flags | FF::MULTISAMPLE_RESOLVE);
        }
    }
    if format == wgpu::TextureFormat::Bgra8Unorm {
        if ctx
            .device_features
            .contains(wgpu::Features::BGRA8UNORM_STORAGE)
        {
            caps.allowed_usages |= TU::STORAGE_BINDING;
            caps.flags |= FF::STORAGE_WRITE_ONLY;
        } else {
            caps.flags.remove(all_storage_flags | FF::STORAGE_ATOMIC);
            caps.allowed_usages
                .remove(TU::STORAGE_BINDING | TU::STORAGE_ATOMIC);
        }
    }

    if caps.flags.intersects(all_multisample_flags)
        && !caps.allowed_usages.contains(TU::RENDER_ATTACHMENT)
    {
        errors.push(format!("Format {format:?} supports multisample but doesn't support being a render attachment, with caps {actual_caps:?}"));
        caps.flags
            .remove(all_multisample_flags | FF::MULTISAMPLE_RESOLVE);
    }
    if caps.flags.intersects(all_storage_flags) != caps.allowed_usages.contains(TU::STORAGE_BINDING)
    {
        errors.push(format!(
            "Format {format:?} has mismatched STORAGE_BINDING support: {actual_caps:?}"
        ));
        caps.flags.remove(all_storage_flags | FF::STORAGE_ATOMIC);
        caps.allowed_usages
            .remove(TU::STORAGE_BINDING | TU::STORAGE_ATOMIC);
    }
    if caps.flags.contains(FF::STORAGE_ATOMIC) != caps.allowed_usages.contains(TU::STORAGE_ATOMIC) {
        errors.push(format!(
            "Format {format:?} has mismatched STORAGE_ATOMIC support: {actual_caps:?}"
        ));
        caps.flags.remove(FF::STORAGE_ATOMIC);
        caps.allowed_usages.remove(TU::STORAGE_ATOMIC);
    }
    if caps.flags.contains(FF::STORAGE_ATOMIC) && !caps.flags.intersects(all_storage_flags) {
        errors.push(format!(
            "Format {format:?} supports STORAGE_ATOMIC but not STORAGE_BINDING: {actual_caps:?}"
        ));
        caps.flags.remove(FF::STORAGE_ATOMIC);
        caps.allowed_usages.remove(TU::STORAGE_ATOMIC);
    }
    if caps.flags.contains(FF::MULTISAMPLE_RESOLVE) && !caps.flags.intersects(all_multisample_flags)
    {
        errors.push(format!(
                    "Format {format:?} supports MULTISAMPLE_RESOLVE but not MULTISAMPLE_*X: {actual_caps:?}"
                ));
        caps.flags.remove(FF::MULTISAMPLE_RESOLVE);
    }
    (actual_caps, caps)
}

/// Smoke test: clear `texture` through a render pass. Single 2D layer, which
/// covers standard/multisample/array (3D isn't render-attachable here).
fn smoke_render_clear(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
) {
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 0,
        array_layer_count: Some(1),
        ..Default::default()
    });
    if format.has_depth_aspect() || format.has_stencil_aspect() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("smoke depth/stencil clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: format.has_depth_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    } else {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("smoke color clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
}

/// Smoke test: render into `msaa` and resolve into a fresh single-sample target.
/// Color only — the core render-pass API has no depth/stencil resolve.
fn smoke_multisample_resolve(
    ctx: &TestingContext,
    encoder: &mut wgpu::CommandEncoder,
    msaa: &wgpu::Texture,
    format: wgpu::TextureFormat,
    dim: u32,
) {
    let resolve = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resolve target"),
        size: wgpu::Extent3d {
            width: dim,
            height: dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 0,
        array_layer_count: Some(1),
        ..Default::default()
    });
    let resolve_view = resolve.create_view(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("smoke multisample resolve"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &msaa_view,
            depth_slice: None,
            resolve_target: Some(&resolve_view),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

/// Smoke test: bind `texture` as a storage texture with the given `access`.
/// No shader/dispatch — the bind group alone validates format/view/usage.
fn smoke_storage_bind(
    ctx: &TestingContext,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
    array_layer_count: Option<u32>,
    access: wgpu::StorageTextureAccess,
) {
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke storage bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access,
                    format,
                    view_dimension,
                },
                count: None,
            }],
        });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(view_dimension),
        array_layer_count,
        ..Default::default()
    });
    let _bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke storage bind group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
}

macro_rules! make_format_feature_functions {
    (
        format_sets: [
            $( ($fmt_name:ident, $fmt_features:expr, $iter:expr,), )*
        ],
        feature_sets: $feature_sets:tt,
    ) => {
        $(
            make_format_feature_functions_internal!(
                format_set: ($fmt_name, $fmt_features, $iter,),
                feature_sets: $feature_sets,
            );
        )*
    };
}

macro_rules! make_format_feature_functions_internal {
    (
        format_set: ($fmt_name:ident, $fmt_features:expr, $iter:expr,),
        feature_sets: [
            $( ($feat_name:ident, $feat_features:expr, $usage_type:expr), )*
        ],
    ) => {
        $(
            paste::paste! {
                #[gpu_test]
                static [<TEXTURE_FORMAT_CAPABILITIES_CHECK_ $fmt_name _ $feat_name>]: GpuTestConfiguration = GpuTestConfiguration::new()
                    .parameters(
                        TestParameters::default()
                            .test_features_limits()
                            .features($fmt_features | $feat_features)
                            .enable_noop()
                            // The exhaustive format/usage matrix allocates a lot
                            // of (small) textures. The DX12 software adapter
                            // (WARP) runs out of memory on it; real GPUs have
                            // ample headroom.
                            .skip(wgpu_test::FailureCase::backend_adapter(
                                wgpu::Backends::DX12,
                                "Microsoft Basic Render Driver",
                            )),
                    )
                    .run_sync(|ctx| {
                        let iter = $iter;
                        texture_format_capabilities_check(ctx, iter, $fmt_features, $feat_features, $usage_type)
                    });
            }
        )*
    };
}

// Not all of these are actually used, namely the compression 3d ones are mostly ignored
make_format_feature_functions!(
    format_sets: [
        (
            compat,
            wgpu::Features::empty(),
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                a.required_features().is_empty()
            })),
        ),
        (
            native,
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                (a.required_features() - wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
                    .is_empty()
            })),
        ),
        (
            norm_16bit,
            wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                a.required_features()
                    .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
                    && (a.required_features()
                        - (wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM))
                        .is_empty()
            })),
        ),
        (
            compression_bc,
            wgpu::Features::TEXTURE_COMPRESSION_BC,
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                a.required_features()
                    .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
                    && (a.required_features()
                        - (wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::TEXTURE_COMPRESSION_BC))
                        .is_empty()
            })),
        ),
        (
            compression_etc2,
            wgpu::Features::TEXTURE_COMPRESSION_ETC2,
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                a.required_features()
                    .contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
                    && (a.required_features()
                        - (wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::TEXTURE_COMPRESSION_ETC2))
                        .is_empty()
            })),
        ),
        (
            compression_astc,
            wgpu::Features::TEXTURE_COMPRESSION_ASTC,
            Box::new([
                wgpu::TextureFormat::Astc {
                    channel: wgpu::AstcChannel::Unorm,
                    block: wgpu::AstcBlock::B4x4,
                },
                wgpu::TextureFormat::Astc {
                    channel: wgpu::AstcChannel::UnormSrgb,
                    block: wgpu::AstcBlock::B4x4,
                },
            ].into_iter()),
        ),
        (
            compression_astc_hdr,
            wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR,
            Box::new(std::iter::once(
                wgpu::TextureFormat::Astc {
                    channel: wgpu::AstcChannel::Hdr,
                    block: wgpu::AstcBlock::B4x4,
                }
            )),
        ),
        (
            nv12,
            wgpu::Features::TEXTURE_FORMAT_NV12,
            Box::new(std::iter::once(wgpu::TextureFormat::NV12)),
        ),
        (
            p010,
            wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            Box::new(std::iter::once(wgpu::TextureFormat::P010)),
        ),
        (
            rg11b10ufloat,
            wgpu::Features::RG11B10UFLOAT_RENDERABLE,
            Box::new(std::iter::once(wgpu::TextureFormat::Rg11b10Ufloat)),
        ),
        (
            depth32floatstencil8,
            wgpu::Features::DEPTH32FLOAT_STENCIL8,
            Box::new(std::iter::once(wgpu::TextureFormat::Depth32FloatStencil8)),
        ),
        (
            r64uint,
            wgpu::Features::TEXTURE_INT64_ATOMIC,
            Box::new(std::iter::once(wgpu::TextureFormat::R64Uint)),
        ),
        (
            bgra8unorm,
            wgpu::Features::BGRA8UNORM_STORAGE,
            Box::new(std::iter::once(wgpu::TextureFormat::Bgra8Unorm)),
        ),
    ],
    feature_sets: [
        (no_features, wgpu::Features::empty(), NORMAL_USAGE_TYPES),
        (atomic, wgpu::Features::TEXTURE_ATOMIC, NORMAL_USAGE_TYPES),
        (compression_bc_3d, wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D, NORMAL_USAGE_TYPES),
        (compression_astc_3d, wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D, NORMAL_USAGE_TYPES),
        (multisample_2d_array, wgpu::Features::MULTISAMPLE_ARRAY, &[UsageType::MultisampleArray]),
    ],
);
