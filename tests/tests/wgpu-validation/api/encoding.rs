//! Tests of [`wgpu::CommandEncoder`] and related.

#[test]
fn as_hal() {
    // Sanity-test that the raw encoding API isn't completely broken.

    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.finish();
}

#[test]
#[should_panic = "Mixing the wgpu encoding API with the raw encoding API is not permitted"]
fn mix_apis_wgpu_then_hal() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&buffer, 0, None);
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
}

#[test]
#[should_panic = "Mixing the wgpu encoding API with the raw encoding API is not permitted"]
fn mix_apis_hal_then_wgpu() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<wgpu_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.clear_buffer(&buffer, 0, None);
}

/// Test that the command encoder’s label is remembered and used in errors.
#[test]
#[should_panic = "In CommandEncoder::finish, label = 'my encoder'"]
fn encoding_error_contains_label_of_encoder() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("my buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("my encoder"),
    });
    // This is erroneous because it is copying to the same buffer.
    encoder.copy_buffer_to_buffer(&buffer, 0, &buffer, 0, 10);
    queue.submit([encoder.finish()]);
}

#[test]
fn multi_draw_indirect_count_offset_alignment() {
    use wgpu_test::{fail, valid};

    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
        ..Default::default()
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
             @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
                .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });

    let encode = |count_offset| {
        let mut encoder = device.create_command_encoder(&Default::default());
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
        pass.multi_draw_indirect_count(&buffer, 0, &buffer, count_offset, 1);
        drop(pass);
        encoder.finish()
    };

    valid(&device, || encode(4));
    fail(
        &device,
        || encode(2),
        Some("Indirect draw count buffer offset 2 is not a multiple of 4"),
    );
}
