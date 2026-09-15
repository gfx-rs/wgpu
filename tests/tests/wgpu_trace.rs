//! Tests of [`wgpu::Buffer`] and related.

use std::any::Any;

use wgpu_core as wgc;
use wgpu_types as wgt;

use wgc::{command::Command, device::trace::Action};

#[derive(Eq, PartialEq)]
enum TestType {
    Normal,
    FailedCommands,
    FailedSubmit,
}

fn trace_test(test_type: TestType) {
    let instance = wgc::instance::Instance::new(
        "test",
        wgt::instance::InstanceDescriptor {
            backends: wgt::Backends::NOOP,
            backend_options: wgt::BackendOptions {
                noop: wgt::NoopBackendOptions::enabled(),
                ..Default::default()
            },
            ..wgt::instance::InstanceDescriptor::new_without_display_handle()
        },
        None,
    );
    let adapter = instance
        .request_adapter(&wgt::RequestAdapterOptions::default(), wgt::Backends::NOOP)
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgt::DeviceDescriptor {
            trace: wgt::Trace::Memory,
            ..Default::default()
        })
        .unwrap();

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = device.create_buffer(&wgt::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgt::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(matches!(device.pop_error_scope(), Ok(None)));

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor::default());
    assert!(matches!(device.pop_error_scope(), Ok(None)));

    match test_type {
        TestType::Normal => {
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            encoder.clear_buffer(buffer, 0, None);
            let cmdbuf = encoder.finish(&wgt::CommandBufferDescriptor::default());
            queue.submit(&[cmdbuf]);
            assert!(matches!(device.pop_error_scope(), Ok(None)));
        }
        TestType::FailedCommands => {
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            // Try to clear past the end of the buffer.
            encoder.clear_buffer(buffer, 0, Some(2048));
            let _cmdbuf = encoder.finish(&wgt::CommandBufferDescriptor::default());
            assert!(matches!(device.pop_error_scope(), Ok(Some(_))));
        }
        TestType::FailedSubmit => {
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            // Destroy the buffer after encoding the clear command, before submitting it.
            encoder.clear_buffer(buffer.clone(), 0, None);
            let cmdbuf = encoder.finish(&wgt::CommandBufferDescriptor::default());
            assert!(matches!(device.pop_error_scope(), Ok(None)));
            buffer.destroy();
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            queue.submit(&[cmdbuf]);
            assert!(matches!(device.pop_error_scope(), Ok(Some(_))));
        }
    }

    let trace = device.take_trace().unwrap();
    let trace = (trace.as_ref() as &dyn Any)
        .downcast_ref::<wgc::device::trace::MemoryTrace>()
        .unwrap();
    let actions = trace.actions();
    dbg!(actions);

    match test_type {
        TestType::Normal => {
            let [.., Action::Submit(_, commands), Action::DropBuffer(_)] = actions else {
                panic!("expected last action to be Submit");
            };
            assert_eq!(commands.len(), 1);
            assert!(matches!(
                commands[0],
                Command::ClearBuffer {
                    dst: _,
                    offset: 0,
                    size: None,
                },
            ));
        }
        TestType::FailedCommands => {
            let Some(Action::FailedCommands {
                commands: Some(commands),
                failed_at_submit: None,
                error,
            }) = actions.last()
            else {
                panic!("expected last action to be FailedCommands");
            };
            assert_eq!(
                error,
                "Clear of 0..2048 would end up overrunning the bounds of the buffer of size 1024"
            );
            assert_eq!(commands.len(), 1);
            assert!(matches!(
                commands[0],
                Command::ClearBuffer {
                    dst: _,
                    offset: 0,
                    size: Some(2048),
                },
            ));
        }
        TestType::FailedSubmit => {
            let Some(Action::FailedCommands {
                commands: Some(commands),
                failed_at_submit: Some(_),
                error,
            }) = actions.last()
            else {
                panic!("expected last action to be FailedCommands");
            };
            assert_eq!(error, "Buffer with '' label has been destroyed");
            assert_eq!(commands.len(), 1);
            assert!(matches!(
                commands[0],
                Command::ClearBuffer {
                    dst: _,
                    offset: 0,
                    size: None,
                },
            ));
        }
    }
}

#[test]
fn trace_clear_buffer() {
    trace_test(TestType::Normal);
}

#[test]
fn trace_failed_commands() {
    trace_test(TestType::FailedCommands);
}

#[test]
fn trace_failed_submit() {
    trace_test(TestType::FailedSubmit);
}

#[test]
fn trace_texture_test() {
    let instance = wgc::instance::Instance::new(
        "test",
        wgt::instance::InstanceDescriptor {
            backends: wgt::Backends::NOOP,
            backend_options: wgt::BackendOptions {
                noop: wgt::NoopBackendOptions::enabled(),
                ..Default::default()
            },
            ..wgt::instance::InstanceDescriptor::new_without_display_handle()
        },
        None,
    );
    let adapter = instance
        .request_adapter(&wgt::RequestAdapterOptions::default(), wgt::Backends::NOOP)
        .unwrap();
    let (device, _) = adapter
        .request_device(&wgt::DeviceDescriptor {
            trace: wgt::Trace::Memory,
            ..Default::default()
        })
        .unwrap();

    let desc = wgt::TextureDescriptor {
        label: None,
        size: wgt::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgt::TextureDimension::D2,
        format: wgt::TextureFormat::Rgba8Unorm,
        usage: wgt::TextureUsages::COPY_DST | wgt::TextureUsages::TEXTURE_BINDING,
        view_formats: Vec::new(),
    };

    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let texture = device.create_texture(&desc);

    assert!(device.pop_error_scope().unwrap().is_none());

    let texture_error = device.create_texture_error(&desc);

    drop(texture);
    drop(texture_error);

    let trace = device.take_trace().unwrap();
    let trace = (trace.as_ref() as &dyn Any)
        .downcast_ref::<wgc::device::trace::MemoryTrace>()
        .unwrap();
    let actions = trace.actions();
    // first one is init
    let actions = &actions[1..];

    assert_eq!(actions.len(), 4);

    let Action::CreateTexture(texture, ..) = actions[0] else {
        panic!("expected first action to be CreateTexture");
    };
    let Action::CreateTextureError(texture_error, ..) = actions[1] else {
        panic!("expected second action to be CreateTextureError");
    };
    let Action::DropTexture(texture_drop) = actions[2] else {
        panic!("expected third action to be DropTexture");
    };
    assert_eq!(texture, texture_drop);
    let Action::DropTexture(texture_error_drop) = actions[3] else {
        panic!("expected fourth action to be DropTexture");
    };
    assert_eq!(texture_error, texture_error_drop);
}
