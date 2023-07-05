use std::num::NonZeroU64;

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandBuffer,
    CommandEncoderDescriptor, ComputePassDescriptor, Maintain, ShaderStages,
};

use wasm_bindgen_test::*;
use wgpu_test::{initialize_test, TestParameters, TestingContext};

fn generate_dummy_work(ctx: &TestingContext) -> CommandBuffer {
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 16,
        usage: BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(16).unwrap()),
                },
                count: None,
            }],
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(buffer.as_entire_buffer_binding()),
        }],
    });

    let mut cmd_buf = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    let mut cpass = cmd_buf.begin_compute_pass(&ComputePassDescriptor::default());
    cpass.set_bind_group(0, &bind_group, &[]);
    drop(cpass);

    cmd_buf.finish()
}

#[test]
#[wasm_bindgen_test]
fn wait() {
    initialize_test(TestParameters::default().skip(), |ctx| {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.device.poll(Maintain::Wait);
    })
}

#[test]
#[wasm_bindgen_test]
fn double_wait() {
    initialize_test(TestParameters::default().skip(), |ctx| {
        let cmd_buf = generate_dummy_work(&ctx);

        ctx.queue.submit(Some(cmd_buf));
        ctx.device.poll(Maintain::Wait);
        ctx.device.poll(Maintain::Wait);
    })
}

#[test]
#[wasm_bindgen_test]
fn wait_on_submission() {
    initialize_test(TestParameters::default().skip(), |ctx| {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index));
    })
}

#[test]
#[wasm_bindgen_test]
fn double_wait_on_submission() {
    initialize_test(TestParameters::default().skip(), |ctx| {
        let cmd_buf = generate_dummy_work(&ctx);

        let index = ctx.queue.submit(Some(cmd_buf));
        ctx.device
            .poll(Maintain::WaitForSubmissionIndex(index.clone()));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index));
    })
}

#[test]
#[wasm_bindgen_test]
fn wait_out_of_order() {
    initialize_test(TestParameters::default().skip(), |ctx| {
        let cmd_buf1 = generate_dummy_work(&ctx);
        let cmd_buf2 = generate_dummy_work(&ctx);

        let index1 = ctx.queue.submit(Some(cmd_buf1));
        let index2 = ctx.queue.submit(Some(cmd_buf2));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index2));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index1));
    })
}
