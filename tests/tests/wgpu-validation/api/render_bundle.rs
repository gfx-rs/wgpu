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
