use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice,
};
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(METAL_EXTERNAL_BUFFER_WRITE);
}

#[gpu_test]
static METAL_EXTERNAL_BUFFER_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().skip(FailureCase::backend(!wgpu::Backends::METAL)))
    .run_sync(|ctx| {
        const SIZE: u64 = 64;
        const WRITTEN: core::ops::Range<usize> = 16..48;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external Metal destination"),
            size: SIZE,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external Metal readback"),
            size: SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let device_ptr = unsafe {
            ctx.device
                .as_hal::<wgpu_hal::api::Metal>()
                .expect("Metal device")
                .retained_raw_handle()
        };
        let queue_ptr = unsafe {
            ctx.queue
                .as_hal::<wgpu_hal::api::Metal>()
                .expect("Metal queue")
                .retained_raw_handle()
        };
        let buffer_ptr = unsafe {
            buffer
                .as_hal::<wgpu_hal::api::Metal>()
                .expect("Metal buffer")
                .retained_raw_handle()
        };

        let device = unsafe {
            Retained::<ProtocolObject<dyn MTLDevice>>::from_raw(device_ptr.cast().as_ptr())
                .expect("retained Metal device")
        };
        let queue = unsafe {
            Retained::<ProtocolObject<dyn MTLCommandQueue>>::from_raw(queue_ptr.cast().as_ptr())
                .expect("retained Metal queue")
        };
        let raw_buffer = unsafe {
            Retained::<ProtocolObject<dyn MTLBuffer>>::from_raw(buffer_ptr.cast().as_ptr())
                .expect("retained Metal buffer")
        };
        assert_eq!(queue.device(), device);

        let command_buffer = queue.commandBuffer().expect("external command buffer");
        let encoder = command_buffer.blitCommandEncoder().expect("blit encoder");
        encoder.fillBuffer_range_value(
            &raw_buffer,
            NSRange::new(WRITTEN.start, WRITTEN.len()),
            0xA5,
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        unsafe {
            buffer
                .mark_external_write_initialized(WRITTEN.start as u64..WRITTEN.end as u64)
                .expect("register completed external write");
        }

        let mut command_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        command_encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, SIZE);
        ctx.queue.submit([command_encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, Result::unwrap);
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = readback.slice(..).get_mapped_range();
        assert!(mapped[..WRITTEN.start].iter().all(|byte| *byte == 0));
        assert!(mapped[WRITTEN.clone()].iter().all(|byte| *byte == 0xA5));
        assert!(mapped[WRITTEN.end..].iter().all(|byte| *byte == 0));
        drop(mapped);
        readback.unmap();
    });
