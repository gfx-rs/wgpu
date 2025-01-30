use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindingResource, ComputePassDescriptor,
    ComputePipelineDescriptor, DownlevelFlags, Extent3d, Features, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static TEXTURE_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT)
            .features(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
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
