use wgpu::util::DeviceExt;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

/// Running a compute shader with one or more of the workgroup sizes set to 0 implies that no work
/// should be done, and is a user error. Vulkan and DX12 accept this invalid input with grace, but
/// Metal does not guard against this and eventually the machine will crash. Since this is a public
/// API that may be given untrusted values in a browser, this must be protected again.
///
/// The following test should successfully do nothing on all platforms.
#[gpu_test]
static ZERO_WORKGROUP_SIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits::default()))
    .run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    "
                @group(0)
                @binding(0)
                var<storage, read_write> vals: array<i32>;

                @compute
                @workgroup_size(1)
                fn main(@builtin(global_invocation_id) id: vec3u) {
                    vals[id.x] = vals[id.x] * i32(id.x);
                }
            ",
                )),
            });
        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let buffer = DeviceExt::create_buffer_init(
            &ctx.device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[1, 1, 1, 1, 1, 1, 1, 1],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        );
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
            cpass.dispatch_workgroups(1, 0, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    });
