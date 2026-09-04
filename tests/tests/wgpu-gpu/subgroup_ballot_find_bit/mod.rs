use std::{borrow::Cow, num::NonZeroU64};

use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(SUBGROUP_BALLOT_FIND_BIT);
}

const THREAD_COUNT: u64 = 64;

// This shader can only be authored in GLSL (or SPIR-V): WGSL has no
// `subgroupBallotFindLSB`/`subgroupBallotFindMSB` built-in, so
// `naga::Statement::SubgroupBallotFindBit` is unreachable from WGSL source.
// Loading it through `ShaderSource::Glsl` exercises naga's GLSL front-end and
// then whichever backend the active adapter actually uses: the native
// `OpGroupNonUniformBallotFindLSB`/`MSB` on Vulkan, native
// `subgroupBallotFindLSB`/`MSB` again on GL, and the formula-based polyfills
// on D3D12 (HLSL) and Metal (MSL).
#[apply(gpu_test!)]
static SUBGROUP_BALLOT_FIND_BIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::SUBGROUP)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        let device = &ctx.device;

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: THREAD_COUNT * size_of::<u32>() as u64,
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
                    min_binding_size: NonZeroU64::new(THREAD_COUNT * size_of::<u32>() as u64),
                },
                count: None,
            }],
        });

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subgroup_ballot_find_bit"),
            source: wgpu::ShaderSource::Glsl {
                shader: Cow::Borrowed(include_str!("shader.comp")),
                stage: naga::ShaderStage::Compute,
                defines: &[],
            },
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
            cpass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        wgpu::util::DownloadBuffer::read_buffer(
            device,
            &ctx.queue,
            &storage_buffer.slice(..),
            |mapping_buffer_view| {
                let mapping_buffer_view = mapping_buffer_view.unwrap();
                let result: &[u32; THREAD_COUNT as usize] =
                    bytemuck::from_bytes(&mapping_buffer_view);
                let expected = [1u32; THREAD_COUNT as usize];
                if result != &expected {
                    let failed: Vec<u32> = result
                        .iter()
                        .enumerate()
                        .filter(|&(_, &v)| v != 1)
                        .map(|(i, _)| i as u32)
                        .collect();
                    panic!(
                        "subgroupBallotFindLSB/MSB mismatch on invocations: {failed:?}\n\
                        got:      {result:?}\n\
                        expected: {expected:?}"
                    );
                }
            },
        );
    });
