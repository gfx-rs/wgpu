use std::time::Duration;
use wgpu::wgt::BufferDescriptor;
use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferUsages,
    ComputePassDescriptor, ComputePipelineDescriptor, DownlevelFlags, Extent3d, MapMode, Origin3d,
    PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static TEXTURE_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT),
    )
    .run_sync(texture_binding);

fn texture_binding(ctx: TestingContext) {
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rg32Float,
        usage: TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
    let bind = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&texture.create_view(&Default::default())),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);
}

#[gpu_test]
static SINGLE_SCALAR_LOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT),
    )
    .run_sync(single_scalar_load);

fn single_scalar_load(ctx: TestingContext) {
    let texture_read = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let texture_write = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: size_of::<[f32; 4]>() as wgpu::BufferAddress,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("single_scalar.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
    let bind = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &texture_write.create_view(&Default::default()),
                ),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(
                    &texture_read.create_view(&Default::default()),
                ),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &texture_write,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(MapMode::Read, move |res| {
        res.unwrap();
        send.send(()).expect("Thread should wait for receive");
    });
    // Poll to run map.
    ctx.device.poll(PollType::Wait).unwrap();
    recv.recv_timeout(Duration::from_secs(10))
        .expect("mapping should not take this long");
    let val = *bytemuck::from_bytes::<[f32; 4]>(&buffer.slice(..).get_mapped_range());
    assert_eq!(val, [0.0, 0.0, 0.0, 1.0]);
}
