use std::{mem::size_of, num::NonZeroU64};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

const THREAD_COUNT: u64 = 128;
const TEST_COUNT: u32 = 32;

#[gpu_test]
static SUBGROUP_OPERATIONS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::SUBGROUP)
            .limits(wgpu::Limits::downlevel_defaults())
            // Expect metal to fail on tests involving operations in divergent control flow
            //
            // Newlines are included in the panic message to ensure that _additional_ failures
            // are not matched against.
            .expect_fail(
                wgpu_test::FailureCase::molten_vk()
                    // 14.3 doesn't fail test 29
                    .panic("thread 0 failed tests: 27,\nthread 1 failed tests: 27, 28,\n")
                    // Prior versions do.
                    .panic("thread 0 failed tests: 27, 29,\nthread 1 failed tests: 27, 28, 29,\n"),
            )
            .expect_fail(
                wgpu_test::FailureCase::backend(wgpu::Backends::METAL)
                    // 14.3 doesn't fail test 29
                    .panic("thread 0 failed tests: 27,\nthread 1 failed tests: 27, 28,\n")
                    // Prior versions do.
                    .panic("thread 0 failed tests: 27, 29,\nthread 1 failed tests: 27, 28, 29,\n"),
            ),
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

        let cs_module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
                let expected_mask = (1u64 << (TEST_COUNT)) - 1; // generate full mask
                let expected_array = [expected_mask as u32; THREAD_COUNT as usize];
                if result != &expected_array {
                    use std::fmt::Write;
                    let mut msg = String::new();
                    writeln!(
                        &mut msg,
                        "Got from GPU:\n{:x?}\n  expected:\n{:x?}",
                        result, &expected_array,
                    )
                    .unwrap();
                    for (thread, (result, expected)) in result
                        .iter()
                        .zip(expected_array)
                        .enumerate()
                        .filter(|(_, (r, e))| *r != e)
                    {
                        write!(&mut msg, "thread {thread} failed tests:").unwrap();
                        let difference = result ^ expected;
                        for i in (0..u32::BITS).filter(|i| (difference & (1 << i)) != 0) {
                            write!(&mut msg, " {i},").unwrap();
                        }
                        writeln!(&mut msg).unwrap();
                    }
                    panic!("{}", msg);
                }
            },
        );
    });
