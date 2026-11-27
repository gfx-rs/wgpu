use wgpu::util::DeviceExt as _;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(ARRAY_LENGTH_WITH_UNUSED_BINDING);
}

/// Get the length of a runtime-sized array when the bind group layout has another
/// storage buffer that the shader does not use. Qualcomm's Adreno 5xx driver crashed
/// recording the dispatch.
#[apply(gpu_test!)]
static ARRAY_LENGTH_WITH_UNUSED_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "
                    @group(0) @binding(0) var<storage, read_write> values: array<u32>;

                    @compute @workgroup_size(1)
                    fn main() {
                        values[0] = arrayLength(&values);
                    }
                    "
                    .into(),
                ),
            });
        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[storage_entry(0), storage_entry(1)],
                });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let values = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0u32; 5]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let unused = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: unused.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values, 0, &readback, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let view = readback.slice(..).get_mapped_range().unwrap();
        let length: &[u32] = bytemuck::cast_slice(&view);
        assert_eq!(length[0], 5);
    });
