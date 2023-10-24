use std::{borrow::Cow, num::NonZeroU64};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

const THREAD_COUNT: u64 = 128;

#[gpu_test]
static SUBGROUP_OPERATIONS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::SUBGROUP_COMPUTE)
            .limits(wgpu::Limits::downlevel_defaults())
            .expect_fail(wgpu_test::FailureCase::molten_vk()),
    )
    .run_sync(|ctx| {
        let device = &ctx.device;

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: THREAD_COUNT * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        THREAD_COUNT * std::mem::size_of::<u32>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(THREAD_COUNT as u32, 1, 1);
        }

        let mapping_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mapping buffer"),
            size: THREAD_COUNT * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &storage_buffer,
            0,
            &mapping_buffer,
            0,
            THREAD_COUNT * std::mem::size_of::<u32>() as u64,
        );
        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        ctx.device.poll(wgpu::Maintain::Wait);
        let mapping_buffer_view = mapping_buffer.slice(..).get_mapped_range();
        let result: &[u32; THREAD_COUNT as usize] = bytemuck::from_bytes(&mapping_buffer_view);
        assert_eq!(result, &[27; THREAD_COUNT as usize]);
    });
