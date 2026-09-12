use std::{borrow::Cow, sync::Arc};

use wgpu_core::{
    command as c,
    device::{queue::Queue, Device},
    instance::Instance,
    resource,
};
use wgpu_types::{error::ErrorFilter, *};

/// The public wgpu render bundle encoder has no debug methods yet, so these tests
/// need a core device rather than the public device provided by the test helpers.
fn core_device(discard_labels: bool) -> (Arc<Device>, Arc<Queue>) {
    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::NOOP;
    desc.flags = InstanceFlags::VALIDATION;
    desc.flags
        .set(InstanceFlags::DISCARD_HAL_LABELS, discard_labels);
    desc.backend_options.noop.enable = true;
    Instance::new("bundle debug validation", desc, None)
        .request_adapter(&Default::default(), Backends::NOOP)
        .unwrap()
        .request_device(&Default::default())
        .unwrap()
}

/// Checks one validation error scope, matching a diagnostic substring or expecting no error.
#[track_caller]
fn check_error<T>(device: &Device, expected: Option<&str>, f: impl FnOnce() -> T) -> T {
    device.push_error_scope(ErrorFilter::Validation);
    let result = f();
    let error = device.pop_error_scope().unwrap();
    assert_eq!(error.is_some(), expected.is_some(), "{error:?}");
    if let Some(expected) = expected {
        assert!(format!("{error:?}").contains(expected), "{error:?}");
    }
    result
}

/// Encodes `+` (push), `-` (pop), and `m` (marker), asserting no encoding-time error.
fn encode(device: &Arc<Device>, ops: &str) -> Box<c::RenderBundleEncoder> {
    let mut encoder = device.create_render_bundle_encoder(&c::RenderBundleEncoderDescriptor {
        color_formats: Cow::Borrowed(&[Some(TextureFormat::Rgba8Unorm)]),
        ..Default::default()
    });
    check_error(device, None, || {
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
        let (device, _queue) = core_device(discard_labels);
        for (ops, expected) in [
            ("", None),
            ("m", None),
            ("+-", None),
            ("++m--", None),
            ("+-+-", None),
            ("-", Some("InvalidPop")),
            ("+", Some("MissingPop")),
            // Equal push/pop counts must not hide an earlier underflow.
            ("-+", Some("InvalidPop")),
            ("+--+", Some("InvalidPop")),
            ("+--", Some("InvalidPop")),
        ] {
            eprintln!("ops={ops:?}, discard_labels={discard_labels}");
            let mut encoder = encode(&device, ops);
            check_error(&device, expected, || encoder.finish(&Default::default()));
        }
    }
}

/// Both successful and failed finish calls permanently end the encoder.
#[test]
fn finish_ends_encoder() {
    let (device, _queue) = core_device(false);
    for (ops, expected) in [
        ("", None),
        ("-", Some("InvalidPop")),
        ("+", Some("MissingPop")),
    ] {
        let mut encoder = encode(&device, ops);
        check_error(&device, expected, || encoder.finish(&Default::default()));
        check_error(&device, Some("Ended"), || encoder.push_debug_group("ended"));
        check_error(&device, Some("Ended"), || encoder.pop_debug_group());
        check_error(&device, Some("Ended"), || {
            encoder.insert_debug_marker("ended")
        });
        check_error(&device, Some("Ended"), || {
            encoder.finish(&Default::default())
        });
    }
}

/// Using an invalid bundle reports an error at command encoder finish, not pass end.
#[test]
fn invalid_bundle_rejected_by_pass() {
    let (device, _queue) = core_device(false);
    let texture = device.create_texture(&resource::TextureDescriptor {
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
        view_formats: vec![],
    });
    for (ops, expected) in [("-", "InvalidPop"), ("+", "MissingPop")] {
        let bundle = check_error(&device, Some(expected), || {
            encode(&device, ops).finish(&Default::default())
        });
        let commands = device.create_command_encoder(&Default::default());
        check_error(&device, None, || {
            let mut pass = commands.begin_render_pass(c::ResolvedRenderPassDescriptor {
                color_attachments: Cow::Owned(vec![Some(c::RenderPassColorAttachment {
                    view: texture.create_view(&Default::default()),
                    depth_slice: None,
                    resolve_target: None,
                    load_op: LoadOp::Clear(Color::BLACK),
                    store_op: StoreOp::Store,
                })]),
                ..Default::default()
            });
            pass.push_debug_group("independent pass group", 0);
            pass.execute_bundles(&[bundle]);
            pass.pop_debug_group();
            pass.end();
        });
        check_error(
            &device,
            Some("RenderBundle with '' label is invalid"),
            || commands.finish(&Default::default()),
        );
    }
}

#[test]
fn dropping_invalid_encoder_does_not_report_error() {
    let (device, _queue) = core_device(false);
    check_error(&device, None, || drop(encode(&device, "-")));
}
