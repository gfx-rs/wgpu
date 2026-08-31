//! Tests that the command buffer records the *first* use of a texture
//! subresource, not its last use, when a pass teaches the command buffer about
//! a subresource it did not know about before.
//!
//! If the first use is wrong, the barriers inserted at submission put the
//! subresource in the wrong layout and the backend reports a validation error.

use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(MIP_FIRST_USE);
}

#[apply(gpu_test!)]
static MIP_FIRST_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits {
        max_storage_textures_per_shader_stage: 1,
        ..Default::default()
    }))
    .run_sync(|ctx| {
        const MIP_LEVELS: u32 = 4;

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test image"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "
                    @group(0) @binding(0) var input: texture_2d<f32>;
                    @group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
                    @compute @workgroup_size(1)
                    fn main() {
                        textureStore(output, vec2u(0u), textureLoad(input, vec2u(0u), 0));
                    }
                    "
                    .into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: None,
                cache: None,
                compilation_options: Default::default(),
            });

        let views: Vec<_> = (0..MIP_LEVELS)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("mip {mip}")),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let bind_group = |read: usize, write: usize| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[read]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[write]),
                    },
                ],
            })
        };

        let read_1_write_0 = bind_group(1, 0);
        let read_1_write_2 = bind_group(1, 2);
        let read_2_write_3 = bind_group(2, 3);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // The first pass leaves mip 2 and mip 3 unknown to the command buffer.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &read_1_write_0, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // The second pass uses mip 2 as a storage target and then as a sampled
        // texture, so its first and last use of mip 2 differ.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &read_1_write_2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_bind_group(0, &read_2_write_3, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    });
