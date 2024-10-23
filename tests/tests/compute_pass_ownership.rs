//! Tests that compute passes take ownership of resources that are associated with.
//! I.e. once a resource is passed in to a compute pass, it can be dropped.

use std::{mem::size_of, num::NonZeroU64};

use wgpu::util::DeviceExt as _;
use wgpu_test::{gpu_test, valid, GpuTestConfiguration, TestParameters, TestingContext};

const SHADER_SRC: &str = "
@group(0) @binding(0)
var<storage, read_write> buffer: array<vec4f>;

@compute @workgroup_size(1, 1, 1) fn main() {
    buffer[0] *= 2.0;
}
";

#[gpu_test]
static COMPUTE_PASS_RESOURCE_OWNERSHIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            // https://github.com/gfx-rs/wgpu/issues/5800
            .skip(wgpu_test::FailureCase::backend_adapter(
                wgpu::Backends::GL,
                "AMD Radeon Pro WX 3200",
            )),
    )
    .run_async(compute_pass_resource_ownership);

async fn compute_pass_resource_ownership(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        indirect_buffer,
        bind_group,
        pipeline,
    } = resource_setup(&ctx);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups_indirect(&indirect_buffer, 0);

        // Now drop all resources we set. Then do a device poll to make sure the resources are really not dropped too early, no matter what.
        drop(pipeline);
        drop(bind_group);
        drop(indirect_buffer);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_compute_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static COMPUTE_PASS_QUERY_SET_OWNERSHIP_PIPELINE_STATISTICS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(wgpu::Features::PIPELINE_STATISTICS_QUERY),
        )
        .run_async(compute_pass_query_set_ownership_pipeline_statistics);

async fn compute_pass_query_set_ownership_pipeline_statistics(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        indirect_buffer: _,
        bind_group,
        pipeline,
    } = resource_setup(&ctx);

    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set"),
        ty: wgpu::QueryType::PipelineStatistics(
            wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
        ),
        count: 1,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.begin_pipeline_statistics_query(&query_set, 0);
        cpass.dispatch_workgroups(1, 1, 1);
        cpass.end_pipeline_statistics_query();

        // Drop the query set. Then do a device poll to make sure it's not dropped too early, no matter what.
        drop(query_set);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_compute_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static COMPUTE_PASS_QUERY_SET_OWNERSHIP_TIMESTAMPS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits().features(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ))
        .run_async(compute_pass_query_set_ownership_timestamps);

async fn compute_pass_query_set_ownership_timestamps(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        indirect_buffer: _,
        bind_group,
        pipeline,
    } = resource_setup(&ctx);

    let query_set_timestamp_writes = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set_timestamp_writes"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_set_write_timestamp = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set_write_timestamp"),
        ty: wgpu::QueryType::Timestamp,
        count: 1,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &query_set_timestamp_writes,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.write_timestamp(&query_set_write_timestamp, 0);
        cpass.dispatch_workgroups(1, 1, 1);

        // Drop the query sets. Then do a device poll to make sure they're not dropped too early, no matter what.
        drop(query_set_timestamp_writes);
        drop(query_set_write_timestamp);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_compute_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static COMPUTE_PASS_KEEP_ENCODER_ALIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(compute_pass_keep_encoder_alive);

async fn compute_pass_keep_encoder_alive(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer: _,
        cpu_buffer: _,
        buffer_size: _,
        indirect_buffer,
        bind_group,
        pipeline,
    } = resource_setup(&ctx);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compute_pass"),
        timestamp_writes: None,
    });

    // Now drop the encoder - it is kept alive by the compute pass.
    // To do so, we have to make the compute pass forget the lifetime constraint first.
    let mut cpass = cpass.forget_lifetime();
    drop(encoder);

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    // Record some draw commands.
    cpass.set_pipeline(&pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.dispatch_workgroups_indirect(&indirect_buffer, 0);

    // Dropping the pass will still execute the pass, even though there's no way to submit it.
    // Ideally, this would log an error, but the encoder is not dropped until the compute pass is dropped,
    // making this a valid operation.
    // (If instead the encoder was explicitly destroyed or finished, this would be an error.)
    valid(&ctx.device, || drop(cpass));
}

async fn assert_compute_pass_executed_normally(
    mut encoder: wgpu::CommandEncoder,
    gpu_buffer: wgpu::Buffer,
    cpu_buffer: wgpu::Buffer,
    buffer_size: u64,
    ctx: TestingContext,
) {
    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let data = cpu_buffer.slice(..).get_mapped_range();

    let floats: &[f32] = bytemuck::cast_slice(&data);
    assert_eq!(floats, [2.0, 4.0, 6.0, 8.0]);
}

// Setup ------------------------------------------------------------

struct ResourceSetup {
    gpu_buffer: wgpu::Buffer,
    cpu_buffer: wgpu::Buffer,
    buffer_size: u64,

    indirect_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

fn resource_setup(ctx: &TestingContext) -> ResourceSetup {
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let buffer_size = 4 * size_of::<f32>() as u64;

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
                    min_binding_size: NonZeroU64::new(buffer_size),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            contents: bytemuck::bytes_of(&[1.0_f32, 2.0, 3.0, 4.0]),
        });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let indirect_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indirect_buffer"),
            usage: wgpu::BufferUsages::INDIRECT,
            contents: wgpu::util::DispatchIndirectArgs { x: 1, y: 1, z: 1 }.as_bytes(),
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

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        indirect_buffer,
        bind_group,
        pipeline,
    }
}
