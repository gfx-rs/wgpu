use super::*;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, TestParameters};

// These tests are split into `COOPERATIVE_MATRIX_F16` and
// `COOPERATIVE_MATRIX_F32` because the latter can run without
// `wgpu::Features::SHADER_F16`.

#[apply(gpu_test!)]
pub static COOPERATIVE_MATRIX_F32: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        // `shader.wgsl` hardcodes 8x8 f32 tiles (`coop_mat8x8<f32, ...>`),
        // so only an exact match is usable here.
        let config = coop_props.iter().find(|prop| {
            prop.m_size == 8
                && prop.n_size == 8
                && prop.k_size == 8
                && prop.ab_type == wgpu::CooperativeScalarType::F32
                && prop.cr_type == wgpu::CooperativeScalarType::F32
        });
        let Some(config) = config else {
            // Not every adapter that supports EXPERIMENTAL_COOPERATIVE_MATRIX
            // exposes an 8x8x8 f32/f32 configuration -- e.g. tensor/matrix-core
            // hardware commonly multiplies in a reduced-precision input type
            // and only optionally accumulates at f32, so plain f32 inputs are
            // often unsupported. We can't `.skip()` this per-adapter without
            // that list growing without bound across every GPU architecture
            // contributors happen to test on, so we log and move on instead.
            log::warn!(
                "No 8x8x8 f32 cooperative matrix configuration found among: \
                 {coop_props:?}; skipping test"
            );
            return;
        };
        let ExecuteResults {
            max_error,
            tolerance,
            matrix: _,
        } = execute(&ctx.device, &ctx.queue, config).await;
        assert!(max_error < tolerance);
    });

#[apply(gpu_test!)]
pub static COOPERATIVE_MATRIX_F16: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        // `shader_f16_16x16.wgsl` hardcodes 16x16 f16 tiles
        // (`coop_mat16x16<f16, ...>`), so only an exact match is usable here.
        let config = coop_props.iter().find(|prop| {
            prop.m_size == 16
                && prop.n_size == 16
                && prop.k_size == 16
                && prop.ab_type == wgpu::CooperativeScalarType::F16
                && prop.cr_type == wgpu::CooperativeScalarType::F16
        });
        let Some(config) = config else {
            // See the comment in COOPERATIVE_MATRIX_F32 above: not every adapter
            // exposes a 16x16x16 f16/f16 configuration (e.g. some only pair
            // smaller tile sizes with f16), and we don't want a per-adapter
            // `.skip()` list to grow without bound, so we log and move on.
            log::warn!(
                "No 16x16x16 f16 cooperative matrix configuration found among: \
                 {coop_props:?}; skipping test"
            );
            return;
        };
        let ExecuteResults {
            max_error,
            tolerance,
            matrix: _,
        } = execute(&ctx.device, &ctx.queue, config).await;
        assert!(max_error < tolerance);
    });

const RECTANGULAR_MATRIX: u32 = 64;

fn scalar_wgsl(ty: wgpu::CooperativeScalarType) -> &'static str {
    match ty {
        wgpu::CooperativeScalarType::F32 => "f32",
        wgpu::CooperativeScalarType::F16 => "f16",
        wgpu::CooperativeScalarType::I32 => "i32",
        wgpu::CooperativeScalarType::U32 => "u32",
    }
}

/// Row-major `C = A * B + C`. Types are `coop_mat{columns}x{rows}`: A is K×M, B is N×K, C is N×M.
fn rectangular_shader(config: &wgpu::CooperativeMatrixProperties, workgroup_size: u32) -> String {
    let ab = scalar_wgsl(config.ab_type);
    let cr = scalar_wgsl(config.cr_type);
    let m = config.m_size;
    let n = config.n_size;
    let k = config.k_size;
    let enable_f16 = if matches!(config.ab_type, wgpu::CooperativeScalarType::F16)
        || matches!(config.cr_type, wgpu::CooperativeScalarType::F16)
    {
        "enable f16;\n"
    } else {
        ""
    };
    format!(
        "{enable_f16}\
enable wgpu_cooperative_matrix;

const N: u32 = {RECTANGULAR_MATRIX}u;
const KDIM: u32 = {RECTANGULAR_MATRIX}u;

@group(0) @binding(0)
var<storage, read> matrix_a: array<{ab}>;
@group(0) @binding(1)
var<storage, read> matrix_b: array<{ab}>;
@group(0) @binding(2)
var<storage, read_write> matrix_c: array<{cr}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {{
    let tile_row = workgroup_id.x * {m}u;
    let tile_col = workgroup_id.y * {n}u;
    let c_offset = tile_row * N + tile_col;
    var c_tile = coopLoadT<coop_mat{n}x{m}<{cr}, C>>(&matrix_c[c_offset], N);

    for (var k: u32 = 0u; k < KDIM; k += {k}u) {{
        let a_offset = tile_row * KDIM + k;
        let a_tile = coopLoadT<coop_mat{k}x{m}<{ab}, A>>(&matrix_a[a_offset], KDIM);
        let b_offset = k * N + tile_col;
        let b_tile = coopLoadT<coop_mat{n}x{k}<{ab}, B>>(&matrix_b[b_offset], N);
        c_tile = coopMultiplyAdd(a_tile, b_tile, c_tile);
    }}

    coopStoreT(c_tile, &matrix_c[c_offset], N);
}}
"
    )
}

fn quantize(value: f32, ty: wgpu::CooperativeScalarType) -> f32 {
    match ty {
        wgpu::CooperativeScalarType::F16 => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}

fn pack_floats(values: &[f32], ty: wgpu::CooperativeScalarType) -> Vec<u8> {
    match ty {
        wgpu::CooperativeScalarType::F16 => {
            let converted: Vec<half::f16> =
                values.iter().copied().map(half::f16::from_f32).collect();
            bytemuck::cast_slice(&converted).to_vec()
        }
        wgpu::CooperativeScalarType::F32 => bytemuck::cast_slice(values).to_vec(),
        wgpu::CooperativeScalarType::I32 | wgpu::CooperativeScalarType::U32 => {
            unreachable!("integer cooperative matrices are not executed by this test")
        }
    }
}

async fn execute_rectangular(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::CooperativeMatrixProperties,
    workgroup_size: u32,
) -> ExecuteResults {
    let dim = RECTANGULAR_MATRIX;
    assert_eq!(dim % config.m_size, 0);
    assert_eq!(dim % config.n_size, 0);
    assert_eq!(dim % config.k_size, 0);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Rectangular Cooperative Matrix Shader"),
        source: wgpu::ShaderSource::Wgsl(rectangular_shader(config, workgroup_size).into()),
    });

    let matrix_a_f32: Vec<f32> = (0..dim * dim)
        .map(|idx| {
            let (i, j) = (idx / dim, idx % dim);
            ((i * 3 + j * 5) % 11) as f32 * 0.1
        })
        .collect();
    let matrix_b_f32: Vec<f32> = (0..dim * dim)
        .map(|idx| {
            let (i, j) = (idx / dim, idx % dim);
            ((i * 7 + j * 11) % 13) as f32 * 0.1
        })
        .collect();
    let matrix_c_f32 = vec![0.0f32; (dim * dim) as usize];

    let bytes_ab = pack_floats(&matrix_a_f32, config.ab_type);
    let bytes_b = pack_floats(&matrix_b_f32, config.ab_type);
    let bytes_c = pack_floats(&matrix_c_f32, config.cr_type);

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix A"),
        size: bytes_ab.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix B"),
        size: bytes_b.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: bytes_c.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: bytes_c.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer_a, 0, &bytes_ab);
    queue.write_buffer(&buffer_b, 0, &bytes_b);
    queue.write_buffer(&buffer_c, 0, &bytes_c);

    let storage = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Rectangular Cooperative Matrix Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Rectangular Cooperative Matrix Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_c.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Rectangular Cooperative Matrix Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Rectangular Cooperative Matrix Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Rectangular Cooperative Matrix Encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Rectangular Cooperative Matrix Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dim / config.m_size, dim / config.n_size, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, staging_buffer.size());
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("Poll failed");
    receiver
        .recv_async()
        .await
        .expect("Channel receive failed")
        .expect("Buffer mapping failed");

    let data = buffer_slice.get_mapped_range().unwrap();
    let result: Vec<f32> = match config.cr_type {
        wgpu::CooperativeScalarType::F16 => {
            let result_f16: Vec<half::f16> = bytemuck::allocation::pod_collect_to_vec(&data);
            result_f16.into_iter().map(|value| value.to_f32()).collect()
        }
        wgpu::CooperativeScalarType::F32 => bytemuck::allocation::pod_collect_to_vec(&data),
        wgpu::CooperativeScalarType::I32 | wgpu::CooperativeScalarType::U32 => unreachable!(),
    };
    drop(data);
    staging_buffer.unmap();

    let mut reference = vec![0.0f32; (dim * dim) as usize];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..dim {
                sum += quantize(matrix_a_f32[(i * dim + k) as usize], config.ab_type)
                    * quantize(matrix_b_f32[(k * dim + j) as usize], config.ab_type);
            }
            reference[(i * dim + j) as usize] = quantize(sum, config.cr_type);
        }
    }

    let tolerance = if config.ab_type == wgpu::CooperativeScalarType::F16
        || config.cr_type == wgpu::CooperativeScalarType::F16
    {
        0.1
    } else {
        0.01
    };
    let mut max_error = 0.0f32;
    for i in 0..(dim * dim) as usize {
        max_error = max_error.max((result[i] - reference[i]).abs());
    }

    ExecuteResults {
        max_error,
        tolerance,
        matrix: result,
    }
}

#[apply(gpu_test!)]
pub static COOPERATIVE_MATRIX_RECTANGULAR: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        let mut seen = std::collections::HashSet::new();
        let configs: Vec<_> = coop_props
            .iter()
            .filter(|prop| {
                let non_square = prop.m_size != prop.n_size || prop.n_size != prop.k_size;
                let float = matches!(
                    (prop.ab_type, prop.cr_type),
                    (
                        wgpu::CooperativeScalarType::F16 | wgpu::CooperativeScalarType::F32,
                        wgpu::CooperativeScalarType::F16 | wgpu::CooperativeScalarType::F32,
                    )
                );
                let key = (
                    prop.m_size,
                    prop.n_size,
                    prop.k_size,
                    prop.ab_type,
                    prop.cr_type,
                );
                non_square && float && !prop.saturating_accumulation && seen.insert(key)
            })
            .copied()
            .collect();
        if configs.is_empty() {
            log::warn!(
                "No non-square f16/f32 cooperative matrix configuration found among: \
                 {coop_props:?}; skipping test"
            );
            return;
        }

        let workgroup_size = ctx.adapter.get_info().subgroup_max_size.clamp(
            1,
            ctx.device
                .limits()
                .max_compute_invocations_per_workgroup
                .max(1),
        );
        for config in configs {
            let ExecuteResults {
                max_error,
                tolerance,
                matrix: _,
            } = execute_rectangular(&ctx.device, &ctx.queue, &config, workgroup_size).await;
            assert!(
                max_error < tolerance,
                "{config:?} max error {max_error} >= tolerance {tolerance}"
            );
        }
    });
