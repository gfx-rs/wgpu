use wgpu::{
    util::DeviceExt, ComputePassTimestampWrites, Features, InstanceFlags,
    QUERY_RESOLVE_BUFFER_ALIGNMENT,
};
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

const SHADER: &str = r#"
@compute @workgroup_size(1)
fn main() {
    return;
}
"#;

const ITERATIONS: u32 = 10;

const QUERIES_PER_ITERATION: u32 = 2;
const TOTAL_QUERIES: u32 = QUERIES_PER_ITERATION * ITERATIONS;

#[gpu_test]
static TIMESTAMP_QUERY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .expect_fail(FailureCase::webgl2())
            .test_features_limits()
            .features(Features::TIMESTAMP_QUERY)
            // Ensure timestamp normalization functions correctly
            .instance_flags(InstanceFlags::AUTOMATIC_TIMESTAMP_NORMALIZATION),
    )
    .run_sync(timestamp_query);

fn timestamp_query(ctx: TestingContext) {
    // Setup pipeline using a simple shader with hardcoded vertices
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("timestamp query shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pipeline"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Create timestamp query set
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Query set"),
        ty: wgpu::QueryType::Timestamp,
        count: TOTAL_QUERIES,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for i in 0..ITERATIONS {
        let base_index = i * QUERIES_PER_ITERATION;

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute pass"),
            timestamp_writes: Some(ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(base_index),
                end_of_pass_write_index: Some(base_index + 1),
            }),
        });
        compute_pass.set_pipeline(&pipeline);

        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    let buffer_size = QUERY_RESOLVE_BUFFER_ALIGNMENT * TOTAL_QUERIES as u64;
    let init_constant = 0x0123_4567_89AB_CDEFu64;

    let init_data = vec![init_constant; buffer_size as usize / 8];

    // Resolve query set to buffer
    let query_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query buffer"),
            contents: bytemuck::cast_slice(&init_data),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        });

    for i in 0..ITERATIONS {
        let start_query = i * QUERIES_PER_ITERATION;
        let end_query = start_query + QUERIES_PER_ITERATION;
        let buffer_offset = i as u64 * QUERY_RESOLVE_BUFFER_ALIGNMENT;

        encoder.resolve_query_set(
            &query_set,
            start_query..end_query,
            &query_buffer,
            buffer_offset,
        );
    }

    let mapping_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mapping buffer"),
        size: query_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&query_buffer, 0, &mapping_buffer, 0, query_buffer.size());

    ctx.queue.submit(Some(encoder.finish()));

    mapping_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| ());
    ctx.device.poll(wgpu::PollType::wait()).unwrap();
    let query_buffer_view = mapping_buffer.slice(..).get_mapped_range();
    let query_data: &[u64] = bytemuck::cast_slice(&query_buffer_view);

    for i in 0..ITERATIONS {
        // The byte and query offset for the current iteration
        let byte_offset = i as u64 * QUERY_RESOLVE_BUFFER_ALIGNMENT;
        let query_offset = byte_offset / 8;

        // The byte and query offset for the next iteration
        let next_byte_offset = (i + 1) as u64 * QUERY_RESOLVE_BUFFER_ALIGNMENT;
        let next_query_offset = next_byte_offset / 8;

        // The range of queries that should still be the value they were initialized to.
        let untouched_query_start = query_offset + QUERIES_PER_ITERATION as u64;
        let untouched_query_end = next_query_offset;

        // WebGPU does not define the value of the timestamp queries. They unfortunately
        // can be `0` in some situations. However, we should expect that some value
        // has been written, and the odds of it being exactly `init_constant` are vanishingly low.
        for query in 0..QUERIES_PER_ITERATION {
            let query_index = query_offset + query as u64;
            assert_ne!(query_data[query_index as usize], init_constant);
        }

        // Validate that the queries that were not written to are still the value they were initialized to.
        for query in untouched_query_start..untouched_query_end {
            assert_eq!(query_data[query as usize], init_constant);
        }
    }
}
