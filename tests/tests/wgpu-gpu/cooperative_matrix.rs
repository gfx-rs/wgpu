use wgpu::util::DeviceExt;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(COOPERATIVE_STORE_RUNTIME_STRIDE);
}

const SENTINEL: f32 = -1.0;

/// `coopStoreT` with a stride read from a uniform writes each row `stride` elements apart.
#[apply(gpu_test!)]
static COOPERATIVE_STORE_RUNTIME_STRIDE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            .limits(wgpu::Limits::default()),
    )
    .run_sync(|ctx| {
        let props = ctx.adapter.cooperative_matrix_properties();
        let Some(size) = props
            .iter()
            .filter(|p| p.cr_type == wgpu::CooperativeScalarType::F32 && p.m_size == p.n_size)
            .map(|p| p.m_size)
            .find(|&size| size == 8 || size == 16)
        else {
            log::warn!("No square f32 accumulator configuration among {props:?}; skipping test");
            return;
        };
        // Vulkan requires the stride to be a multiple of 16 bytes (VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986).
        let stride = size + 4;

        let source = format!(
            "enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> stride: u32;
@compute @workgroup_size(64)
fn main() {{
    let m = coopLoadT<coop_mat{size}x{size}<f32, C>>(&src[0], {size}u);
    coopStoreT(m, &dst[0], stride);
}}"
        );
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let n = size as usize;
        let input: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let src = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dst_len = n * stride as usize;
        let dst = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vec![SENTINEL; dst_len]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let uniform = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&stride),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        wgpu::util::DownloadBuffer::read_buffer(
            &ctx.device,
            &ctx.queue,
            &dst.slice(..),
            move |view| {
                let view = view.unwrap();
                let output: &[f32] = bytemuck::cast_slice(&view);
                let expected: Vec<f32> = (0..dst_len)
                    .map(|i| {
                        let (row, col) = (i / stride as usize, i % stride as usize);
                        if col < n {
                            (row * n + col) as f32
                        } else {
                            SENTINEL
                        }
                    })
                    .collect();
                assert_eq!(
                    output,
                    &expected[..],
                    "{size}x{size} store with stride {stride}"
                );
            },
        );
    });
