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
    let global = wgc::global::Global::new(
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
    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions::default(),
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();
    let (device_id, queue_id) = global
        .adapter_request_device(
            adapter_id,
            &wgt::DeviceDescriptor {
                trace: wgt::Trace::Memory,
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap();

    let (buffer_id, error) = global.device_create_buffer(
        device_id,
        &wgt::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgt::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
        None,
    );
    assert!(error.is_none());

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor::default(),
        None,
    );
    assert!(error.is_none());

    match test_type {
        TestType::Normal => {
            global
                .command_encoder_clear_buffer(encoder_id, buffer_id, 0, None)
                .unwrap();
            let (cmdbuf_id, error) = global.command_encoder_finish(
                encoder_id,
                &wgt::CommandBufferDescriptor::default(),
                None,
            );
            assert!(error.is_none());
            global.queue_submit(queue_id, &[cmdbuf_id]).unwrap();
        }
        TestType::FailedCommands => {
            // Try to clear past the end of the buffer.
            global
                .command_encoder_clear_buffer(encoder_id, buffer_id, 0, Some(2048))
                .unwrap();
            let (_cmdbuf_id, error) = global.command_encoder_finish(
                encoder_id,
                &wgt::CommandBufferDescriptor::default(),
                None,
            );
            assert!(error.is_some());
        }
        TestType::FailedSubmit => {
            // Destroy the buffer after encoding the clear command, before submitting it.
            global
                .command_encoder_clear_buffer(encoder_id, buffer_id, 0, None)
                .unwrap();
            let (cmdbuf_id, error) = global.command_encoder_finish(
                encoder_id,
                &wgt::CommandBufferDescriptor::default(),
                None,
            );
            assert!(error.is_none());
            global.buffer_destroy(buffer_id);
            global.queue_submit(queue_id, &[cmdbuf_id]).unwrap_err();
        }
    }

    let trace = global.device_take_trace(device_id).unwrap();
    let trace = (trace.as_ref() as &dyn Any)
        .downcast_ref::<wgc::device::trace::MemoryTrace>()
        .unwrap();
    let actions = trace.actions();

    match test_type {
        TestType::Normal => {
            let Some(Action::Submit(_, commands)) = actions.last() else {
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
