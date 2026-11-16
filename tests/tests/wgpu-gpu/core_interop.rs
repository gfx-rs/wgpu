//! Tests for sharing one wgpu-core device between several `wgpu` wrappers
//! through `as_core` / `from_core`.
//!
//! This is how an embedder lets a WebGPU binding built directly on wgpu-core
//! (for example `deno_webgpu`) and a renderer written against `wgpu` use the
//! same device, queue and textures without copying through the CPU.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use wgpu::Backends;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        SHARED_TEXTURE_OUTLIVES_PRODUCER,
        SHARED_DEVICE_SHARES_ERROR_SCOPES,
        SHARED_DEVICE_SHARES_LOST_CALLBACK,
    ]);
}

const WIDTH: u32 = 256;
const HEIGHT: u32 = 2;

/// A texture created and written through one wrapper is adopted by a second
/// wrapper of the same wgpu-core device. The producer drops its handle before
/// the consumer copies the texture out through its own device and queue: the
/// texture must stay alive and keep its contents and descriptor.
#[apply(gpu_test!)]
static SHARED_TEXTURE_OUTLIVES_PRODUCER: GpuTestConfiguration = GpuTestConfiguration::new()
    // `as_core` / `from_core` need the wgpu-core backend.
    .parameters(TestParameters::default().skip(FailureCase::backend(Backends::BROWSER_WEBGPU)))
    .run_async(shared_texture_outlives_producer);

async fn shared_texture_outlives_producer(ctx: TestingContext) {
    let core_device = ctx.device.as_core().unwrap();
    let consumer_device = wgpu::Device::from_core(core_device.clone());
    let consumer_queue = wgpu::Queue::from_core(ctx.queue.as_core().unwrap());

    // Both wrappers refer to the very same wgpu-core objects.
    assert!(Arc::ptr_eq(
        &consumer_device.as_core().unwrap(),
        &core_device
    ));
    assert!(Arc::ptr_eq(
        &consumer_queue.as_core().unwrap(),
        &ctx.queue.as_core().unwrap()
    ));

    let size = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let producer_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("producer texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let expected: Vec<u8> = (0..WIDTH * HEIGHT).map(|i| (i % 251) as u8).collect();
    ctx.queue.write_texture(
        producer_texture.as_image_copy(),
        &expected,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH),
            rows_per_image: Some(HEIGHT),
        },
        size,
    );
    ctx.queue.submit(None);

    let consumer_texture = wgpu::Texture::from_core(producer_texture.as_core().unwrap());
    assert_eq!(consumer_texture.size(), producer_texture.size());
    assert_eq!(consumer_texture.format(), producer_texture.format());
    assert_eq!(consumer_texture.usage(), producer_texture.usage());
    assert_eq!(consumer_texture.dimension(), producer_texture.dimension());
    assert_eq!(
        consumer_texture.mip_level_count(),
        producer_texture.mip_level_count()
    );
    assert_eq!(
        consumer_texture.sample_count(),
        producer_texture.sample_count()
    );

    // The producer releases its handle; the consumer's handle keeps the
    // texture alive.
    drop(producer_texture);

    let read_buffer = consumer_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("consumer readback"),
        size: (WIDTH * HEIGHT) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        consumer_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        consumer_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH),
                rows_per_image: Some(HEIGHT),
            },
        },
        size,
    );
    consumer_queue.submit(Some(encoder.finish()));

    let slice = read_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
    // Polling either wrapper polls the shared wgpu-core device.
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();
    assert_eq!(data, expected);
}

/// Error scopes and the uncaptured error handler belong to the wgpu-core
/// device, not to a `wgpu::Device` wrapper: a validation error raised through
/// one wrapper is captured by a scope pushed through another wrapper on the
/// same thread.
#[apply(gpu_test!)]
static SHARED_DEVICE_SHARES_ERROR_SCOPES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .enable_noop()
            .skip(FailureCase::backend(Backends::BROWSER_WEBGPU)),
    )
    .run_async(|ctx| async move {
        let other = wgpu::Device::from_core(ctx.device.as_core().unwrap());

        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        // A zero-sized texture is a validation error.
        let _texture = other.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 0,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        assert!(
            scope.pop().await.is_some(),
            "the scope pushed through one wrapper should capture an error raised through another"
        );
    });

/// The device-lost callback is a single slot of the wgpu-core device. Setting
/// it through a second wrapper replaces the callback set through the first.
#[apply(gpu_test!)]
static SHARED_DEVICE_SHARES_LOST_CALLBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .enable_noop()
            .skip(FailureCase::backend(Backends::BROWSER_WEBGPU)),
    )
    .run_async(|ctx| async move {
        static FIRST_CALLED: AtomicBool = AtomicBool::new(false);
        static SECOND_CALLED: AtomicBool = AtomicBool::new(false);

        let other = wgpu::Device::from_core(ctx.device.as_core().unwrap());
        ctx.device
            .set_device_lost_callback(|_, _| FIRST_CALLED.store(true, Ordering::SeqCst));
        other.set_device_lost_callback(|reason, _| {
            assert!(matches!(reason, wgpu::DeviceLostReason::Destroyed));
            SECOND_CALLED.store(true, Ordering::SeqCst);
        });

        // Destroying through either wrapper destroys the shared device.
        ctx.device.destroy();
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        assert!(SECOND_CALLED.load(Ordering::SeqCst));
        assert!(
            !FIRST_CALLED.load(Ordering::SeqCst),
            "the callback set through the first wrapper should have been replaced"
        );
    });
