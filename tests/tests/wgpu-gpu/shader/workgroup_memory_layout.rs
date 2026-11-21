use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ComputePipelineDescriptor,
    DownlevelFlags, Limits, MapMode, PipelineCompilationOptions, PollType,
};
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(WORKGROUP_MEMORY_LAYOUT);
}

const WORKGROUPS: u32 = 8;
const OUTPUT_SIZE: u64 = WORKGROUPS as u64 * 4 * 4;

/// Workgroup variables with mixed types and alignments, used by several entry
/// points in one pass, are zero-initialized and keep their values.
#[apply(gpu_test!)]
static WORKGROUP_MEMORY_LAYOUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../../../naga/tests/in/wgsl/workgroup-memory-layout.wgsl")
                        .into(),
                ),
            });

        let cases = [
            ("main", 513),
            ("padded_only", 0),
            ("no_workgroup", 0),
            ("main", 7),
            ("padded_only", 0),
            ("main", 513),
        ];

        let mut dispatches = Vec::new();
        for &(entry_point, n) in &cases {
            let constants: &[(&str, f64)] = if entry_point == "main" {
                &[("N", f64::from(n))]
            } else {
                &[]
            };
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: PipelineCompilationOptions {
                        constants,
                        zero_initialize_workgroup_memory: true,
                    },
                    cache: None,
                });
            let output = ctx.device.create_buffer(&BufferDescriptor {
                label: Some(entry_point),
                size: OUTPUT_SIZE,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                }],
            });
            dispatches.push((pipeline, output, bind_group));
        }

        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: OUTPUT_SIZE * cases.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (pipeline, _, bind_group) in &dispatches {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(WORKGROUPS, 1, 1);
            }
        }
        for (index, (_, output, _)) in dispatches.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                output,
                0,
                &readback,
                index as u64 * OUTPUT_SIZE,
                OUTPUT_SIZE,
            );
        }
        ctx.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        for (index, &(entry_point, n)) in cases.iter().enumerate() {
            for group in 0..WORKGROUPS {
                let expected = match entry_point {
                    "main" => [0, n * (n - 1) / 2 + n * group, 64, 136],
                    "padded_only" => [0, 58, 0, 0],
                    _ => [group, 0, 0, 0],
                };
                let offset = index * (OUTPUT_SIZE as usize / 4) + group as usize * 4;
                assert_eq!(
                    words[offset..offset + 4],
                    expected,
                    "case {index} ({entry_point}, N = {n}), workgroup {group}"
                );
            }
        }
    });
