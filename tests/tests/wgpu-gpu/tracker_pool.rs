//! Tests for the device's pool of trackers.
//!
//! Once a submission completes, `wgpu-core` keeps the trackers of its command buffers, and the
//! trackers of finished compute passes, and gives them to later command encoders and passes.

use wgpu::util::DeviceExt as _;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        POOLED_TRACKERS_RELEASE_RESOURCES,
        POOLED_TRACKERS_RECORD_CORRECT_WORK,
        POOLED_TRACKERS_SERVE_ENCODERS_THAT_USE_NOTHING,
    ]);
}

const SHADER_SRC: &str = "
@group(0) @binding(0)
var<storage, read_write> value: u32;

@compute @workgroup_size(1) fn main() {
    value += 1u;
}
";

fn create_pipeline(ctx: &TestingContext) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Records and submits a compute pass, a render pass with an occlusion query and a copy, then
/// drops every resource they use.
fn submit_work(ctx: &TestingContext) {
    let pipeline = create_pipeline(ctx);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let copy_dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("copy_dst"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set"),
        ty: wgpu::QueryType::Occlusion,
        count: 1,
    });
    let query_resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("query_resolve"),
        size: wgpu::QUERY_SIZE.into(),
        usage: wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations::default(),
            })],
            occlusion_query_set: Some(&query_set),
            ..Default::default()
        });
        rpass.begin_occlusion_query(0);
        rpass.end_occlusion_query();
    }
    encoder.resolve_query_set(&query_set, 0..1, &query_resolve, 0);
    encoder.copy_buffer_to_buffer(&buffer, 0, &copy_dst, 0, 4);
    ctx.queue.submit([encoder.finish()]);
}

/// A tracker in the pool holds no resources: once the work that used them has completed,
/// dropped resources are destroyed.
#[apply(gpu_test!)]
static POOLED_TRACKERS_RELEASE_RESOURCES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            // NOTE: The WebGPU backend does not implement internal counters.
            .skip(FailureCase::backend(wgpu::Backends::BROWSER_WEBGPU)),
    )
    .run_async(|ctx| async move {
        // The first round fills the pool and creates any resources wgpu creates lazily.
        submit_work(&ctx);
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let before = ctx.device.get_internal_counters().hal;

        // The second round records with trackers from the pool.
        submit_work(&ctx);
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let after = ctx.device.get_internal_counters().hal;

        assert_eq!(after.buffers.read(), before.buffers.read(), "buffers");
        assert_eq!(after.textures.read(), before.textures.read(), "textures");
        assert_eq!(
            after.texture_views.read(),
            before.texture_views.read(),
            "texture views"
        );
        assert_eq!(
            after.bind_groups.read(),
            before.bind_groups.read(),
            "bind groups"
        );
        assert_eq!(
            after.compute_pipelines.read(),
            before.compute_pipelines.read(),
            "compute pipelines"
        );
        assert_eq!(
            after.query_sets.read(),
            before.query_sets.read(),
            "query sets"
        );
    });

/// Command buffers recorded with trackers from the pool produce the same results as ones
/// recorded with new trackers, while the device's resource count grows past the size the
/// pooled trackers were last used with.
#[apply(gpu_test!)]
static POOLED_TRACKERS_RECORD_CORRECT_WORK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(|ctx| async move {
        let pipeline = create_pipeline(&ctx);
        let value = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("value"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: value.as_entire_binding(),
            }],
        });

        // Each round keeps its buffer alive, so the next round's buffer takes a new tracker index
        // and raises the device's resource count.
        let mut readbacks = Vec::new();
        for expected in 1..=4u32 {
            let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&value, 0, &readback, 0, 4);
            ctx.queue.submit([encoder.finish()]);

            readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
            {
                let data = readback.slice(..).get_mapped_range().unwrap();
                assert_eq!(bytemuck::from_bytes::<u32>(&data), &expected);
            }
            readbacks.push(readback);
        }
    });

/// A tracker from the pool serves a command encoder and a compute pass that use no resources
/// of the kinds it held before.
#[apply(gpu_test!)]
static POOLED_TRACKERS_SERVE_ENCODERS_THAT_USE_NOTHING: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_async(|ctx| async move {
            submit_work(&ctx);
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let empty_encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            ctx.queue.submit([encoder.finish(), empty_encoder.finish()]);
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
        });
