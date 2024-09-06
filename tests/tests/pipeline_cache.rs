use std::{fmt::Write, num::NonZeroU64};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

/// We want to test that using a pipeline cache doesn't cause failure
///
/// It would be nice if we could also assert that reusing a pipeline cache would make compilation
/// be faster however, some drivers use a fallback pipeline cache, which makes this inconsistent
/// (both intra- and inter-run).
#[gpu_test]
static PIPELINE_CACHE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::PIPELINE_CACHE),
    )
    .run_async(pipeline_cache_test);

/// Set to a higher value if adding a timing based assertion. This is otherwise fast to compile
const ARRAY_SIZE: u64 = 256;

/// Create a shader which should be slow-ish to compile
fn shader() -> String {
    let mut body = String::new();
    for idx in 0..ARRAY_SIZE {
        // "Safety": There will only be a single workgroup, and a single thread in that workgroup
        writeln!(body, "    output[{idx}] = {idx}u;")
            .expect("`u64::fmt` and `String::write_fmt` are infallible");
    }

    format!(
        r#"
        @group(0) @binding(0)
        var<storage, read_write> output: array<u32>;

        @compute @workgroup_size(1)
        fn main() {{
        {body}
        }}
        "#,
    )
}

async fn pipeline_cache_test(ctx: TestingContext) {
    let shader = shader();
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(ARRAY_SIZE * 4),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let first_cache_data;
    {
        let first_cache = unsafe {
            ctx.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("pipeline_cache"),
                    data: None,
                    fallback: false,
                })
        };
        let first_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Some(&first_cache),
            });
        validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;
        first_cache_data = first_cache.get_data();
    }
    assert!(first_cache_data.is_some());

    let second_cache = unsafe {
        ctx.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipeline_cache"),
                data: first_cache_data.as_deref(),
                fallback: false,
            })
    };
    let first_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Some(&second_cache),
        });
    validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;

    // Ideally, we could assert here that the second compilation was faster than the first
    // However, that doesn't actually work, because drivers have their own internal caches.
    // This does work on my machine if I set `MESA_DISABLE_PIPELINE_CACHE=1`
    // before running the test; but of course that is not a realistic scenario
}

async fn validate_pipeline(
    ctx: &TestingContext,
    pipeline: wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gpu_buffer: &wgpu::Buffer,
    cpu_buffer: &wgpu::Buffer,
) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, Some(bind_group), &[]);

        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(gpu_buffer, 0, cpu_buffer, 0, ARRAY_SIZE * 4);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let data = cpu_buffer.slice(..).get_mapped_range();

    let arrays: &[u32] = bytemuck::cast_slice(&data);

    assert_eq!(arrays.len(), ARRAY_SIZE as usize);
    for (idx, value) in arrays.iter().copied().enumerate() {
        assert_eq!(value as usize, idx);
    }
    drop(data);
    cpu_buffer.unmap();
}
