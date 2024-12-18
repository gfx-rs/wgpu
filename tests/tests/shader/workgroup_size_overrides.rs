use std::mem::size_of_val;
use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages, Maintain, MapMode};
use wgpu_test::{fail_if, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

const SHADER: &str = r#"
    override n = 3;

    @group(0) @binding(0)
    var<storage, read_write> output: array<u32>;

    @compute @workgroup_size(n - 2)
    fn main(@builtin(local_invocation_index) lii: u32) {
        output[lii] = lii + 2;
    }
"#;

#[gpu_test]
static WORKGROUP_SIZE_OVERRIDES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits::default()))
    .run_async(move |ctx| async move {
        workgroup_size_overrides(&ctx, None, &[2, 0, 0], false).await;
        workgroup_size_overrides(&ctx, Some(4), &[2, 3, 0], false).await;
        workgroup_size_overrides(&ctx, Some(1), &[0, 0, 0], true).await;
    });

async fn workgroup_size_overrides(
    ctx: &TestingContext,
    n: Option<u32>,
    out: &[u32],
    should_fail: bool,
) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
    let pipeline_options = wgpu::PipelineCompilationOptions {
        constants: &[("n".to_owned(), n.unwrap_or(0).into())].into(),
        ..Default::default()
    };
    let compute_pipeline = fail_if(
        &ctx.device,
        should_fail,
        || {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: if n.is_some() {
                        pipeline_options
                    } else {
                        wgpu::PipelineCompilationOptions::default()
                    },
                    cache: None,
                })
        },
        None,
    );
    if should_fail {
        return;
    }
    let init: &[u32] = &[0, 0, 0];
    let init_size: u64 = size_of_val(init).try_into().unwrap();
    let buffer = DeviceExt::create_buffer_init(
        &ctx.device,
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        },
    );
    let mapping_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("mapping buffer"),
        size: init_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_entries = [wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &mapping_buffer, 0, init_size);
    ctx.queue.submit(Some(encoder.finish()));

    mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(Maintain::wait()).await.panic_on_timeout();

    let mapped = mapping_buffer.slice(..).get_mapped_range();

    let typed: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(typed, out);
}
