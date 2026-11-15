//! Tests of [`wgpu::Backend::Noop`].

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

#[test]
fn device_is_not_available_by_default() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect_err("noop backend adapter present when it should not be");
}

#[test]
fn device_is_available_when_requested() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be");
}

#[test]
fn device_and_buffers() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    // Demonstrate that creating and *writing* to a buffer succeeds.
    // This also involves creation of a staging buffer.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hello world"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    assert_eq!(buffer.size(), 8);
    queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]);
    queue.write_buffer(&buffer, 4, &[5, 6, 7, 8]);

    // Demonstrate that we can read back data from the buffer.
    // This also involves copy_buffer_to_buffer().
    let done: Arc<AtomicBool> = Arc::default();
    let done2 = done.clone();
    wgpu::util::DownloadBuffer::read_buffer(&device, &queue, &buffer.slice(..), move |result| {
        assert_eq!(*result.unwrap(), [1, 2, 3, 4, 5, 6, 7, 8],);
        done.store(true, Relaxed);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(done2.load(Relaxed));
}

/// Validated indirect draws do not panic on noop.
#[test]
fn draw_indirect_with_indirect_validation() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
                .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });
    let view = texture.create_view(&Default::default());
    let indirect = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: Default::default(),
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.draw_indirect(&indirect, 0);
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
