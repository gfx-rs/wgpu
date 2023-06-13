use std::num::NonZeroU64;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoderDescriptor, ComputePassDescriptor,
    Maintain, ShaderStages,
};

use wgpu_test::{infra::GpuTest, TestingContext};

struct DummyWorkData {
    _buffer: Buffer,
    _bgl: BindGroupLayout,
    _bg: BindGroup,
    cmd_buf: CommandBuffer,
}

impl DummyWorkData {
    fn new(ctx: &TestingContext) -> Self {
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

        Self {
            _buffer: buffer,
            _bgl: bind_group_layout,
            _bg: bind_group,
            cmd_buf: cmd_buf.finish(),
        }
    }
}

#[derive(Default)]
pub struct WaitTest;

impl GpuTest for WaitTest {
    fn run(&self, ctx: TestingContext) {
        let data = DummyWorkData::new(&ctx);

        ctx.queue.submit(Some(data.cmd_buf));
        ctx.device.poll(Maintain::Wait);
    }
}

#[derive(Default)]
pub struct DoubleWaitTest;

impl GpuTest for DoubleWaitTest {
    fn run(&self, ctx: TestingContext) {
        let data = DummyWorkData::new(&ctx);

        ctx.queue.submit(Some(data.cmd_buf));
        ctx.device.poll(Maintain::Wait);
        ctx.device.poll(Maintain::Wait);
    }
}

#[derive(Default)]
pub struct WaitOnSubmissionTest;

impl GpuTest for WaitOnSubmissionTest {
    fn run(&self, ctx: TestingContext) {
        let data = DummyWorkData::new(&ctx);

        let index = ctx.queue.submit(Some(data.cmd_buf));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index));
    }
}

#[derive(Default)]
pub struct DoubleWaitOnSubmissionTest;

impl GpuTest for DoubleWaitOnSubmissionTest {
    fn run(&self, ctx: TestingContext) {
        let data = DummyWorkData::new(&ctx);

        let index = ctx.queue.submit(Some(data.cmd_buf));
        ctx.device
            .poll(Maintain::WaitForSubmissionIndex(index.clone()));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index));
    }
}

#[derive(Default)]
pub struct WaitOutOfOrderTest;

impl GpuTest for WaitOutOfOrderTest {
    fn run(&self, ctx: TestingContext) {
        let data1 = DummyWorkData::new(&ctx);
        let data2 = DummyWorkData::new(&ctx);

        let index1 = ctx.queue.submit(Some(data1.cmd_buf));
        let index2 = ctx.queue.submit(Some(data2.cmd_buf));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index2));
        ctx.device.poll(Maintain::WaitForSubmissionIndex(index1));
    }
}
