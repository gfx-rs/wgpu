use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};
use wgt::{Backend, Backends};

/// Tests that writing and reading to the max length of a container (vec, mat, array)
/// in the workgroup, private and function address spaces + let declarations
/// will instead write to and read from the last element.
#[gpu_test]
static RESTRICT_WORKGROUP_PRIVATE_FUNCTION_LET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
            .skip(FailureCase::backend(Backends::GL)),
    )
    .run_async(|ctx| async move {
        let test_resources = TestResources::new(&ctx);

        ctx.queue
            .write_buffer(&test_resources.in_buffer, 0, bytemuck::bytes_of(&4_u32));

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&test_resources.pipeline);
            compute_pass.set_bind_group(0, &test_resources.bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &test_resources.out_buffer,
            0,
            &test_resources.readback_buffer,
            0,
            12 * 4,
        );

        ctx.queue.submit(Some(encoder.finish()));

        test_resources
            .readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

        let view = test_resources.readback_buffer.slice(..).get_mapped_range();

        let current_res: [u32; 12] = *bytemuck::from_bytes(&view);
        drop(view);
        test_resources.readback_buffer.unmap();

        if ctx.adapter_info.backend == Backend::Dx12 {
            assert_eq!([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0], current_res);
        } else {
            assert_eq!([1; 12], current_res);
        }
    });

struct TestResources {
    pipeline: wgpu::ComputePipeline,
    in_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TestResources {
    fn new(ctx: &TestingContext) -> Self {
        // FXC doesn't support dynamically indexing and writing to vectors and matrices, it errors with:
        // error X3500: array reference cannot be used as an l-value; not natively addressable
        // see also: https://github.com/gfx-rs/wgpu/issues/4460
        let opt = if ctx.adapter_info.backend == Backend::Dx12 {
            "//"
        } else {
            ""
        };
        let shader_src = format!(
            "
            @group(0) @binding(0)
            var<storage, read_write> in: u32;
            @group(0) @binding(1)
            var<storage, read_write> out: array<u32>;

            var<workgroup> wg_array: array<u32, 3>;
            var<workgroup> wg_vec: vec3u;
            var<workgroup> wg_mat: mat3x3f;

            var<private> private_array: array<u32, 3>;
            var<private> private_vec: vec3u;
            var<private> private_mat: mat3x3f;

            @compute @workgroup_size(1)
            fn main() {{
                let i = in;

                var var_array = array<u32, 3>();
                wg_array[i] = 1u;
                private_array[i] = 1u;
                var_array[i] = 1u;
                let let_array = var_array;

                out[0] = wg_array[i];
                out[1] = private_array[i];
                out[2] = var_array[i];
                out[3] = let_array[i];

                var var_vec = vec3u();
                wg_vec[i] = 1u;
                {opt} private_vec[i] = 1u;
                {opt} var_vec[i] = 1u;
                let let_vec = var_vec;

                out[4] = wg_vec[i];
                out[5] = private_vec[i];
                out[6] = var_vec[i];
                out[7] = let_vec[i];

                var var_mat = mat3x3f();
                wg_mat[i][0] = 1f;
                {opt} private_mat[i][0] = 1f;
                {opt} var_mat[i][0] = 1f;
                let let_mat = var_mat;

                out[8] = u32(wg_mat[i][0]);
                out[9] = u32(private_mat[i][0]);
                out[10] = u32(var_mat[i][0]);
                out[11] = u32(let_mat[i][0]);
            }}
        "
        );

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgt::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgt::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
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

        let in_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            in_buffer,
            out_buffer,
            readback_buffer,
            bind_group,
        }
    }
}
