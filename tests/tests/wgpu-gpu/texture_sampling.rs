use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(CUBE_SAMPLE_NAN_COORDINATES);
}

/// Sampling a cube texture with a direction that has a NaN component must stay
/// within the bounds of the texture and select the expected face.
///
/// Cube face selection chooses a face from the largest-magnitude component of
/// the direction and divides the other two components by it. A NaN component
/// can defeat the face selection on some drivers, which then compute face
/// coordinates outside of the texture.
///
/// The direction `(1, 0, NaN)` has its NaN replaced with zero by the SPIR-V
/// backend, leaving `(1, 0, 0)`, which samples the center texel of the +X face.
/// That texel is given a distinct color from the rest of the cube, so the test
/// only passes if the sample lands exactly there.
///
/// This fix lives in the SPIR-V backend, so the test is gated to Vulkan.
#[apply(gpu_test!)]
static CUBE_SAMPLE_NAN_COORDINATES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::backend(
                wgpu::Backends::all().difference(wgpu::Backends::VULKAN),
            )),
    )
    .run_async(test_impl);

const FACE_SIZE: u32 = 5;
/// The color of every texel except the one the sample should land on.
const BACKGROUND: [u8; 4] = [16, 32, 48, 64];
/// The color of the center texel of the +X face.
const TARGET: [u8; 4] = [192, 160, 128, 224];

async fn test_impl(ctx: TestingContext) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "
                @group(0) @binding(0) var<storage, read_write> io: array<f32, 8>;
                @group(0) @binding(1) var tex: texture_cube<f32>;
                @group(0) @binding(2) var samp: sampler;

                @compute @workgroup_size(1)
                fn main() {
                    let dir = vec3f(io[0], io[1], io[2]);
                    let color = textureSampleLevel(tex, samp, dir, 0.0);
                    io[4] = color.x;
                    io[5] = color.y;
                    io[6] = color.z;
                    io[7] = color.w;
                }
                "
                .into(),
            ),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: FACE_SIZE,
            height: FACE_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Fill the whole cube with the background color, then recolor the center
    // texel of the +X face (layer 0).
    let mut texel_data = BACKGROUND.repeat((FACE_SIZE * FACE_SIZE * 6) as usize);
    let center = FACE_SIZE / 2;
    let center_offset = ((center * FACE_SIZE + center) * 4) as usize;
    texel_data[center_offset..center_offset + 4].copy_from_slice(&TARGET);

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texel_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(FACE_SIZE * 4),
            rows_per_image: Some(FACE_SIZE),
        },
        wgpu::Extent3d {
            width: FACE_SIZE,
            height: FACE_SIZE,
            depth_or_array_layers: 6,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Positive X dominates, so this selects the center of the +X face once the
    // NaN is sanitized to zero.
    let direction = [1.0f32, 0.0, f32::NAN];
    let mut io = [0.0f32; 8];
    io[..3].copy_from_slice(&direction);

    let io_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of_val(&io) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&io_buffer, 0, bytemuck::cast_slice(&io));

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of_val(&io) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: io_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&io_buffer, 0, &readback_buffer, 0, size_of_val(&io) as u64);
    ctx.queue.submit(Some(encoder.finish()));

    readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let result: [f32; 8] = {
        let view = readback_buffer.slice(..).get_mapped_range().unwrap();
        *bytemuck::from_bytes(&view)
    };
    readback_buffer.unmap();

    for (channel, &expected) in TARGET.iter().enumerate() {
        let expected = expected as f32 / 255.0;
        let actual = result[4 + channel];
        assert!(
            (actual - expected).abs() < 1.0 / 255.0,
            "channel {channel} sampled {actual}, expected {expected}; \
             the cube sample did not land on the +X face center texel"
        );
    }
}
