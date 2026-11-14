use wgpu::*;
use wgpu_test::{fail, fail_if, valid};

/// Encodes `+` (push), `-` (pop), and `m` (marker), asserting no encoding-time error.
fn encode<'a>(device: &'a Device, ops: &str) -> RenderBundleEncoder<'a> {
    let mut encoder = device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
        color_formats: &[Some(TextureFormat::Rgba8Unorm)],
        sample_count: 1,
        ..Default::default()
    });
    valid(device, || {
        for op in ops.chars() {
            match op {
                '+' => encoder.push_debug_group("group\0\u{1f31e}"),
                '-' => encoder.pop_debug_group(),
                'm' => encoder.insert_debug_marker(""),
                _ => unreachable!(),
            }
        }
    });
    encoder
}

/// Group errors are reported at finish, even when HAL labels are discarded.
#[test]
fn debug_group_balance() {
    for discard_labels in [false, true] {
        let params = wgpu_test::TestParameters::default();
        let params = if discard_labels {
            params.instance_flags(InstanceFlags::DISCARD_HAL_LABELS)
        } else {
            params.remove_instance_flags(InstanceFlags::DISCARD_HAL_LABELS)
        };
        let instance = wgpu_test::initialize_instance(Backends::NOOP, &params);
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        for (ops, expected) in [
            ("", None),
            ("m", None),
            ("+-", None),
            ("++m--", None),
            ("+-+-", None),
            ("-", Some("Cannot pop debug group")),
            ("+", Some("A debug group was not popped")),
            // Equal push/pop counts must not hide an earlier underflow.
            ("-+", Some("Cannot pop debug group")),
            ("+--+", Some("Cannot pop debug group")),
            ("+--", Some("Cannot pop debug group")),
        ] {
            eprintln!("ops={ops:?}, discard_labels={discard_labels}");
            let encoder = encode(&device, ops);
            fail_if(
                &device,
                expected.is_some(),
                || encoder.finish(&Default::default()),
                expected,
            );
        }
    }
}

/// Both successful and failed finish calls permanently end the encoder.
/// Uses core directly because the public Rust API consumes the encoder at finish.
#[test]
fn finish_ends_encoder() {
    use wgpu_core::{command as c, instance::Instance};

    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::NOOP;
    desc.backend_options.noop.enable = true;
    let (device, _queue) = Instance::new("bundle debug validation", desc, None)
        .request_adapter(&Default::default(), Backends::NOOP)
        .unwrap()
        .request_device(&Default::default())
        .unwrap();
    for op in [None, Some('-'), Some('+')] {
        let mut encoder = device.create_render_bundle_encoder(&c::RenderBundleEncoderDescriptor {
            color_formats: std::borrow::Cow::Borrowed(&[Some(TextureFormat::Rgba8Unorm)]),
            ..Default::default()
        });
        match op {
            Some('-') => encoder.pop_debug_group(),
            Some('+') => encoder.push_debug_group("unclosed"),
            _ => {}
        }
        device.push_error_scope(ErrorFilter::Validation);
        encoder.finish(&Default::default());
        assert_eq!(device.pop_error_scope().unwrap().is_some(), op.is_some());
        for operation in ["push", "pop", "marker", "finish"] {
            device.push_error_scope(ErrorFilter::Validation);
            match operation {
                "push" => encoder.push_debug_group("ended"),
                "pop" => encoder.pop_debug_group(),
                "marker" => encoder.insert_debug_marker("ended"),
                "finish" => {
                    encoder.finish(&Default::default());
                }
                _ => unreachable!(),
            }
            let error = device.pop_error_scope().unwrap().unwrap();
            let expected = if operation == "finish" {
                "Render bundle encoder has already ended"
            } else {
                "Encoding must not have ended"
            };
            assert!(
                error.to_string().contains(expected),
                "{operation} after {op:?}: {error}"
            );
        }
    }
}

/// Using an invalid bundle reports an error at command encoder finish, not pass end.
#[test]
fn invalid_bundle_rejected_by_pass() {
    let (device, _queue) = Device::noop(&Default::default());
    let texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });
    let view = texture.create_view(&Default::default());
    for (ops, expected) in [
        ("-", "Cannot pop debug group"),
        ("+", "A debug group was not popped"),
    ] {
        let bundle = fail(
            &device,
            || encode(&device, ops).finish(&Default::default()),
            Some(expected),
        );
        let mut commands = device.create_command_encoder(&Default::default());
        valid(&device, || {
            let mut pass = commands.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                ..Default::default()
            });
            pass.push_debug_group("independent pass group");
            pass.execute_bundles([&bundle]);
            pass.pop_debug_group();
        });
        fail(
            &device,
            || commands.finish(),
            Some("RenderBundle with '' label is invalid"),
        );
    }
}

#[test]
fn dropping_invalid_encoder_does_not_report_error() {
    let (device, _queue) = Device::noop(&Default::default());
    valid(&device, || drop(encode(&device, "-")));
}

#[test]
fn bundle_resource_lifetimes() {
    for api in ["inherent", "bundle trait object", "pass trait object"] {
        eprintln!("{api}");
        // The noop adapter's limits do not satisfy indirect draw validation.
        let params = wgpu_test::TestParameters::default()
            .remove_instance_flags(InstanceFlags::VALIDATION_INDIRECT_CALL);
        let instance = wgpu_test::initialize_instance(Backends::NOOP, &params);
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        valid(&device, || {
            let texture = device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let mut commands = device.create_command_encoder(&Default::default());
            let mut pass = commands.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                ..Default::default()
            });
            let mut bundle = encode(&device, "");
            {
                let shader = device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::Wgsl(
                        "@group(0) @binding(0) var<uniform> position: vec4f;
                        @vertex fn vs(@location(0) vertex: vec4f) -> @builtin(position) vec4f {
                            return position + vertex;
                        }
                        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
                            .into(),
                    ),
                });
                let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                    label: None,
                    layout: None,
                    vertex: VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(VertexBufferLayout {
                            array_stride: 16,
                            step_mode: VertexStepMode::Vertex,
                            attributes: &vertex_attr_array![0 => Float32x4],
                        })],
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(TextureFormat::Rgba8Unorm.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
                let uniform = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                let vertex = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 48,
                    usage: BufferUsages::VERTEX,
                    mapped_at_creation: false,
                });
                let index = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 6,
                    usage: BufferUsages::INDEX,
                    mapped_at_creation: false,
                });
                let indirect = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 36,
                    usage: BufferUsages::INDIRECT,
                    mapped_at_creation: false,
                });
                if api == "inherent" {
                    bundle.set_bind_group(0, None, &[]);
                    bundle.set_vertex_buffer(0, None);
                    bundle.set_pipeline(&pipeline);
                    bundle.set_bind_group(0, &bind_group, &[]);
                    bundle.set_vertex_buffer(0, vertex.slice(..));
                    bundle.set_index_buffer(index.slice(..), IndexFormat::Uint16);
                    bundle.draw_indirect(&indirect, 0);
                    bundle.draw_indexed_indirect(&indirect, 16);
                } else {
                    let encoder: &mut dyn util::RenderEncoder<'_> = if api == "bundle trait object"
                    {
                        &mut bundle
                    } else {
                        &mut pass
                    };
                    encoder.set_bind_group(0, None, &[]);
                    encoder.set_vertex_buffer(0, None);
                    encoder.set_pipeline(&pipeline);
                    encoder.set_bind_group(0, Some(&bind_group), &[]);
                    encoder.set_vertex_buffer(0, Some(vertex.slice(..)));
                    encoder.set_index_buffer(index.slice(..), IndexFormat::Uint16);
                    encoder.draw_indirect(&indirect, 0);
                    encoder.draw_indexed_indirect(&indirect, 16);
                }
            }
            let bundle = bundle.finish(&Default::default());
            if api != "pass trait object" {
                pass.execute_bundles([&bundle]);
            }
            drop(pass);
            queue.submit([commands.finish()]);
        });
    }
}

#[test]
fn draw_errors_reported_at_finish() {
    let (device, _queue) = Device::noop(&Default::default());
    let mut bundle = encode(&device, "");
    valid(&device, || bundle.draw(0..3, 0..1));
    fail(
        &device,
        || bundle.finish(&Default::default()),
        Some("Pipeline must be set"),
    );

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
                .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(TextureFormat::Bgra8Unorm.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let mut bundle = encode(&device, "");
    valid(&device, || bundle.set_pipeline(&pipeline));
    fail(
        &device,
        || bundle.finish(&Default::default()),
        Some("Incompatible color attachments"),
    );
}
