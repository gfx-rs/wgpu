use exhaust::Exhaust;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

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
    usage_types: &[UsageType],
) {
    use wgpu::TextureFormatFeatureFlags as FF;
    use wgpu::TextureUsages as TU;

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
            let can_3d = if format.has_depth_aspect()
                || format.has_stencil_aspect()
                || format.is_multi_planar_format()
            {
                false
            } else if format.is_astc() {
                ctx.device_features
                    .contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
            } else if format.is_bcn() {
                ctx.device_features
                    .contains(wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
            } else {
                !format.is_compressed()
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
                    if !caps.flags.intersects(all_multisample_flags)
                        || !ctx
                            .device_features
                            .contains(wgpu::Features::MULTISAMPLE_ARRAY)
                    {
                        continue;
                    }
                }
                UsageType::Transient => {
                    caps.allowed_usages &= TU::TRANSIENT_ATTACHMENT | TU::RENDER_ATTACHMENT;
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
                    // Rendering into a 3D texture needs a `depth_slice`, which the
                    // smoke tests don't use.
                    caps.allowed_usages
                        .remove(TU::RENDER_ATTACHMENT | TU::TRANSIENT_ATTACHMENT);
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
                caps.allowed_usages.remove(TU::TRANSIENT_ATTACHMENT);
            }

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
                texture_binding_view_dimension: None,
            };
            let texture1 = ctx.device.create_texture(&texture_desc);

            for &(other_format, aspect) in &view_formats_aspects {
                let (other_actual_caps, other_caps) = get_caps(&ctx, other_format, &mut Vec::new());
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
                    swizzle: wgpu::TextureComponentSwizzle::default(),
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
            for access in storage_accesses(&ctx, caps) {
                eprintln!("Using it as a {access:?} storage texture");
                smoke_storage(
                    &ctx,
                    &mut encoder,
                    &texture1,
                    format,
                    view_dim,
                    array_layer_count,
                    access,
                );
            }
            // Exercise *every* sample count the format claims, not just the
            // highest one chosen for `texture1`. Each count gets a fresh
            // texture cloned from the same descriptor.
            for ms_count in supported_multisample_counts(caps.flags) {
                eprintln!("Rendering into it multisampled (sample_count {ms_count})");
                let ms_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    sample_count: ms_count,
                    ..texture_desc.clone()
                });
                smoke_render_clear(&mut encoder, &ms_texture, format);
                if caps.flags.contains(FF::MULTISAMPLE_RESOLVE)
                    && !format.has_depth_aspect()
                    && !format.has_stencil_aspect()
                {
                    eprintln!("Resolving it (sample_count {ms_count})");
                    smoke_multisample_resolve(&ctx, &mut encoder, &ms_texture, format, TEXTURE_DIM);
                }
            }
        }
        // Submit and then poll immediately. This lets the textures be dropped to avoid OOMs with combinatorial explosion
        eprintln!("Attempting to record & submit command buffer");
        ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        if !errors.is_empty() {
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

/// Every multisample count a format advertises via its
/// `MULTISAMPLE_X{2,4,8,16}` flags, smallest first. Used so the smoke tests
/// exercise *each* claimed count instead of only the highest one.
fn supported_multisample_counts(flags: wgpu::TextureFormatFeatureFlags) -> Vec<u32> {
    use wgpu::TextureFormatFeatureFlags as FF;
    let mut counts = Vec::new();
    for (flag, count) in [
        (FF::MULTISAMPLE_X2, 2),
        (FF::MULTISAMPLE_X4, 4),
        (FF::MULTISAMPLE_X8, 8),
        (FF::MULTISAMPLE_X16, 16),
    ] {
        if flags.contains(flag) {
            counts.push(count);
        }
    }
    counts
}

/// Smoke test: clear `texture` through a render pass. Single 2D layer, which
/// covers standard/multisample/array (3D isn't render-attachable here).
fn smoke_render_clear(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
) {
    // Multi-planar formats can only be rendered to through per-plane views.
    if let Some(planes) = format.planes() {
        for plane in 0..planes {
            let aspect = wgpu::TextureAspect::from_plane(plane).unwrap();
            let plane_format = format.aspect_specific_format(aspect).unwrap();
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(plane_format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect,
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            smoke_render_clear_view(encoder, &view, texture.usage(), plane_format);
        }
        return;
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 0,
        array_layer_count: Some(1),
        ..Default::default()
    });
    smoke_render_clear_view(encoder, &view, texture.usage(), format);
}

fn smoke_render_clear_view(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
) {
    // TRANSIENT (memoryless) attachments can't be stored, only discarded.
    let store = if usage.contains(wgpu::TextureUsages::TRANSIENT_ATTACHMENT) {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    };
    if format.has_depth_aspect() || format.has_stencil_aspect() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("smoke depth/stencil clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: format.has_depth_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store,
                }),
                stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store,
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
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store,
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
        texture_binding_view_dimension: None,
    });
    let msaa_view = msaa.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: 0,
        array_layer_count: Some(1),
        ..Default::default()
    });
    let resolve_view = resolve.create_view(&Default::default());
    // The resolve target always receives the resolved output; a memoryless
    // (TRANSIENT) multisample source can only discard its own contents.
    let store = if msaa
        .usage()
        .contains(wgpu::TextureUsages::TRANSIENT_ATTACHMENT)
    {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    };
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("smoke multisample resolve"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &msaa_view,
            depth_slice: None,
            resolve_target: Some(&resolve_view),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

/// Every storage access mode that `caps` claims.
fn storage_accesses(
    ctx: &TestingContext,
    caps: wgpu::TextureFormatFeatures,
) -> Vec<wgpu::StorageTextureAccess> {
    use wgpu::StorageTextureAccess as Sta;
    use wgpu::TextureFormatFeatureFlags as FF;
    use wgpu::TextureUsages as TU;

    let mut accesses = Vec::new();
    if !caps.allowed_usages.contains(TU::STORAGE_BINDING) {
        return accesses;
    }
    for (flag, access) in [
        (FF::STORAGE_READ_ONLY, Sta::ReadOnly),
        (FF::STORAGE_WRITE_ONLY, Sta::WriteOnly),
        (FF::STORAGE_READ_WRITE, Sta::ReadWrite),
    ] {
        if caps.flags.contains(flag) {
            accesses.push(access);
        }
    }
    // Atomic access also needs `TEXTURE_ATOMIC`, even when adapter-specific
    // features report `STORAGE_ATOMIC`.
    if caps.allowed_usages.contains(TU::STORAGE_ATOMIC)
        && ctx.device_features.contains(wgpu::Features::TEXTURE_ATOMIC)
    {
        accesses.push(Sta::Atomic);
    }
    accesses
}

/// The WGSL texel format name of `format`, if WGSL can declare a storage
/// texture with it.
fn wgsl_storage_format(format: wgpu::TextureFormat) -> Option<String> {
    const WGSL_STORAGE_FORMATS: &[&str] = &[
        "r8unorm",
        "r8snorm",
        "r8uint",
        "r8sint",
        "r16unorm",
        "r16snorm",
        "r16uint",
        "r16sint",
        "r16float",
        "rg8unorm",
        "rg8snorm",
        "rg8uint",
        "rg8sint",
        "r32uint",
        "r32sint",
        "r32float",
        "rg16unorm",
        "rg16snorm",
        "rg16uint",
        "rg16sint",
        "rg16float",
        "rgba8unorm",
        "rgba8snorm",
        "rgba8uint",
        "rgba8sint",
        "rgb10a2uint",
        "rgb10a2unorm",
        "rg11b10ufloat",
        "r64uint",
        "rg32uint",
        "rg32sint",
        "rg32float",
        "rgba16unorm",
        "rgba16snorm",
        "rgba16uint",
        "rgba16sint",
        "rgba16float",
        "rgba32uint",
        "rgba32sint",
        "rgba32float",
        "bgra8unorm",
    ];
    let name = serde_json::to_value(format).ok()?.as_str()?.to_owned();
    WGSL_STORAGE_FORMATS
        .contains(&name.as_str())
        .then_some(name)
}

/// Smoke test: bind `texture` as a storage texture with the given `access`,
/// and dispatch a compute shader that accesses it that way.
fn smoke_storage(
    ctx: &TestingContext,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
    array_layer_count: Option<u32>,
    access: wgpu::StorageTextureAccess,
) {
    use wgpu::StorageTextureAccess as Sta;
    use wgpu::TextureViewDimension as Tvd;

    // Storage textures are color-only; a whole-image view of a combined
    // depth/stencil format has both aspects, which can't be a storage binding.
    if format.has_depth_aspect() || format.has_stencil_aspect() {
        return;
    }
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
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke storage bind group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });

    let Some(texel_format) = wgsl_storage_format(format) else {
        return;
    };
    let (dim, coords) = match view_dimension {
        Tvd::D2 => ("2d", "vec2<u32>(0u)"),
        Tvd::D2Array => ("2d_array", "vec2<u32>(0u), 0u"),
        Tvd::D3 => ("3d", "vec3<u32>(0u)"),
        _ => unreachable!(),
    };
    let scalar = match format.sample_type(None, None).unwrap() {
        _ if format == wgpu::TextureFormat::R64Uint => "u64",
        wgpu::TextureSampleType::Float { .. } => "f32",
        wgpu::TextureSampleType::Uint => "u32",
        wgpu::TextureSampleType::Sint => "i32",
        wgpu::TextureSampleType::Depth => unreachable!(),
    };
    let (access_name, body) = match access {
        Sta::ReadOnly => ("read", format!("_ = textureLoad(t, {coords});")),
        Sta::WriteOnly => (
            "write",
            format!("textureStore(t, {coords}, vec4<{scalar}>());"),
        ),
        Sta::ReadWrite => (
            "read_write",
            format!("textureStore(t, {coords}, textureLoad(t, {coords}));"),
        ),
        Sta::Atomic => (
            "atomic",
            format!("textureAtomicMax(t, {coords}, {scalar}(1));"),
        ),
    };
    let source = format!(
        "@group(0) @binding(0) var t: texture_storage_{dim}<{texel_format}, {access_name}>;\n\
         @compute @workgroup_size(1) fn main() {{ {body} }}"
    );
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke storage shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("smoke storage pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(1, 1, 1);
}

macro_rules! make_format_feature_functions {
    (
        format_sets: [
            $( ($fmt_name:ident, $fmt_features:expr, $iter:expr, [ $($extra:tt)* ],), )*
        ],
        feature_sets: $common:tt,
    ) => {
        $(
            make_format_feature_functions_internal!(
                format_set: ($fmt_name, $fmt_features, $iter,),
                feature_sets: $common,
                extra_feature_sets: [ $($extra)* ],
            );
        )*

        pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
            paste::paste! {
                $( [<all_tests_ $fmt_name>](tests); )*
            }
        }
    };
}

macro_rules! make_format_feature_functions_internal {
    (
        format_set: $format_set:tt,
        feature_sets: [ $($common:tt)* ],
        extra_feature_sets: [ $($extra:tt)* ],
    ) => {
        make_format_feature_functions_internal!(
            format_set: $format_set,
            feature_sets: [ $($common)* $($extra)* ],
        );
    };
    (
        format_set: ($fmt_name:ident, $fmt_features:expr, $iter:expr,),
        feature_sets: [
            $( ($feat_name:ident, $feat_features:expr, $usage_type:expr), )*
        ],
    ) => {
        paste::paste! {
            fn [<all_tests_ $fmt_name>](tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
                $( tests.push([<TEXTURE_FORMAT_CAPABILITIES_CHECK_ $fmt_name _ $feat_name>]); )*
            }
        }
        $(
            paste::paste! {
                #[apply(gpu_test!)]
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
                        texture_format_capabilities_check(ctx, iter, $fmt_features | $feat_features, $usage_type)
                    });
            }
        )*
    };
}

make_format_feature_functions!(
    format_sets: [
        (
            compat,
            wgpu::Features::empty(),
            Box::new(wgpu::TextureFormat::exhaust().filter(|a| {
                a.required_features().is_empty()
            })),
            [
                (atomic, wgpu::Features::TEXTURE_ATOMIC, NORMAL_USAGE_TYPES),
                (adapter_specific_atomic, wgpu::Features::TEXTURE_ATOMIC | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, NORMAL_USAGE_TYPES),
            ],
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
            [],
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
            [
                (compression_bc_3d, wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D, NORMAL_USAGE_TYPES),
                (adapter_specific_compression_bc_3d, wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, NORMAL_USAGE_TYPES),
            ],
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
            [],
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
            [
                (compression_astc_3d, wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D, NORMAL_USAGE_TYPES),
                (adapter_specific_compression_astc_3d, wgpu::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, NORMAL_USAGE_TYPES),
            ],
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
            [],
        ),
        (
            nv12,
            wgpu::Features::TEXTURE_FORMAT_NV12,
            Box::new(std::iter::once(wgpu::TextureFormat::NV12)),
            [],
        ),
        (
            p010,
            wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            Box::new(std::iter::once(wgpu::TextureFormat::P010)),
            [],
        ),
        (
            rg11b10ufloat,
            wgpu::Features::RG11B10UFLOAT_RENDERABLE,
            Box::new(std::iter::once(wgpu::TextureFormat::Rg11b10Ufloat)),
            [],
        ),
        (
            depth32floatstencil8,
            wgpu::Features::DEPTH32FLOAT_STENCIL8,
            Box::new(std::iter::once(wgpu::TextureFormat::Depth32FloatStencil8)),
            [],
        ),
        (
            r64uint,
            // `SHADER_INT64` is needed to access the texture from a shader.
            wgpu::Features::TEXTURE_INT64_ATOMIC | wgpu::Features::SHADER_INT64,
            Box::new(std::iter::once(wgpu::TextureFormat::R64Uint)),
            [
                (atomic, wgpu::Features::TEXTURE_ATOMIC, NORMAL_USAGE_TYPES),
                (adapter_specific_atomic, wgpu::Features::TEXTURE_ATOMIC | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, NORMAL_USAGE_TYPES),
            ],
        ),
        (
            bgra8unorm,
            wgpu::Features::BGRA8UNORM_STORAGE,
            Box::new(std::iter::once(wgpu::TextureFormat::Bgra8Unorm)),
            [],
        ),
    ],
    feature_sets: [
        (no_features, wgpu::Features::empty(), NORMAL_USAGE_TYPES),
        (adapter_specific, wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, NORMAL_USAGE_TYPES),
        (multisample_2d_array, wgpu::Features::MULTISAMPLE_ARRAY, &[UsageType::MultisampleArray]),
        (adapter_specific_multisample_2d_array, wgpu::Features::MULTISAMPLE_ARRAY | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, &[UsageType::MultisampleArray]),
    ],
);
