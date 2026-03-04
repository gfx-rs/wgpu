//! Cooperative Matrix Multiplication Example
//!
//! This example demonstrates how to use cooperative matrix operations
//! (also known as tensor cores on NVIDIA GPUs or simdgroup matrix
//! operations on Apple GPUs) to perform efficient matrix multiplication.
//!
//! Cooperative matrices allow a workgroup to collectively load, store,
//! and perform matrix operations on small tiles of data, enabling
//! hardware-accelerated matrix math.
//!
//! Note: This feature requires hardware support and is currently
//! experimental. Use `adapter.cooperative_matrix_properties()` to query
//! supported configurations:
//! - Metal (Apple): 8x8 f32, 8x8 f16, mixed precision (f16 inputs, f32 accumulator)
//! - Vulkan (AMD): Typically 16x16 f16
//! - Vulkan (NVIDIA): Varies by GPU generation

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Matrix dimensions for our example (must be divisible by tile size)
const M: u32 = 64; // Rows of A and C
const N: u32 = 64; // Cols of B and C
const K: u32 = 64; // Cols of A, Rows of B

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dimensions {
    m: u32,
    n: u32,
    k: u32,
    stride: u32,
}

async fn run() {
    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Failed to find an appropriate adapter");

    log::info!("Using adapter: {:?}", adapter.get_info());

    // Query supported cooperative matrix configurations
    let coop_props = adapter.cooperative_matrix_properties();
    if coop_props.is_empty() {
        log::error!(
            "Cooperative matrix is not supported on this adapter.\n\
            This feature requires:\n\
            - Metal: Apple7+ (A14/M1) with MSL 2.3+\n\
            - Vulkan: VK_KHR_cooperative_matrix extension"
        );
        return;
    }

    // Display supported configurations
    log::info!("Supported cooperative matrix configurations:");
    for (i, prop) in coop_props.iter().enumerate() {
        log::info!(
            "  [{}] {:?}x{:?}x{:?} - AB: {:?}, CR: {:?}{}",
            i,
            prop.m_size,
            prop.n_size,
            prop.k_size,
            prop.ab_type,
            prop.cr_type,
            if prop.saturating_accumulation {
                " (saturating)"
            } else {
                ""
            }
        );
    }

    // Find a suitable configuration - prefer f32, but accept f16
    // Try 16x16 first (AMD), then 8x8 (Apple Metal)
    let selected_config = coop_props
        .iter()
        .find(|prop| {
            prop.m_size == 16
                && prop.n_size == 16
                && prop.k_size == 16
                && prop.ab_type == wgpu::CooperativeScalarType::F16
                && prop.cr_type == wgpu::CooperativeScalarType::F16
        })
        .or_else(|| {
            coop_props.iter().find(|prop| {
                prop.m_size == 8
                    && prop.n_size == 8
                    && prop.k_size == 8
                    && prop.ab_type == wgpu::CooperativeScalarType::F32
                    && prop.cr_type == wgpu::CooperativeScalarType::F32
            })
        });

    let config = match selected_config {
        Some(c) => {
            log::info!(
                "Selected configuration: {:?}x{:?}x{:?} AB={:?} CR={:?}",
                c.m_size,
                c.n_size,
                c.k_size,
                c.ab_type,
                c.cr_type
            );
            c
        }
        None => {
            log::error!(
                "No suitable cooperative matrix configuration found.\n\
                This example supports 16x16 f16 (AMD) or 8x8 f32 (Apple Metal).\n\
                Available configurations are listed above."
            );
            return;
        }
    };

    let tile_size = config.m_size;
    let use_f16 = config.ab_type == wgpu::CooperativeScalarType::F16;

    log::info!(
        "Using {}x{} tiles with {} precision",
        tile_size,
        tile_size,
        if use_f16 { "f16" } else { "f32" }
    );

    // Check if cooperative matrix is supported
    let adapter_features = adapter.features();
    if !adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        log::error!("EXPERIMENTAL_COOPERATIVE_MATRIX feature not available");
        return;
    }

    // Check if f16 is needed and available
    if use_f16 && !adapter_features.contains(wgpu::Features::SHADER_F16) {
        log::error!("SHADER_F16 feature not available, but required for f16 cooperative matrices");
        return;
    }

    // Build required features
    let mut required_features = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    if use_f16 {
        required_features |= wgpu::Features::SHADER_F16;
    }

    // Request device with experimental features enabled
    let (device, queue) = unsafe {
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Cooperative Matrix Device"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device")
    };

    let results = execute(&device, &queue, config).await;

    log::info!(
        "Matrix multiplication {M}x{K}x{N} completed using {} precision!",
        if use_f16 { "f16" } else { "f32" }
    );
    log::info!("Max error vs CPU reference: {:.6}", results.max_error);

    if results.max_error < results.tolerance {
        log::info!(
            "✓ Results match CPU reference within tolerance ({})",
            results.tolerance
        );
    } else {
        log::warn!(
            "✗ Results differ from CPU reference (tolerance: {})",
            results.tolerance
        );
    }

    // Print a small sample of the result
    log::info!("Sample of result matrix C (top-left 4x4):");
    for i in 0..4 {
        let row: Vec<String> = (0..4)
            .map(|j| format!("{:6.2}", results.matrix[i * N as usize + j]))
            .collect();
        log::info!("  [{}]", row.join(", "));
    }
}

struct ExecuteResults {
    max_error: f32,
    tolerance: f32,
    matrix: Vec<f32>,
}

async fn execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::CooperativeMatrixProperties,
) -> ExecuteResults {
    let use_f16 = config.ab_type == wgpu::CooperativeScalarType::F16;

    // Select the appropriate shader based on configuration
    let shader_source = if use_f16 {
        include_str!("shader_f16_16x16.wgsl")
    } else {
        include_str!("shader.wgsl")
    };

    // Create the shader module using the standard validated path
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cooperative Matrix Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Initialize matrices
    // A is MxK, B is KxN, C is MxN (result)
    // Use f32 for computation, convert to f16 if needed for GPU
    let matrix_a_f32: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let matrix_b_f32: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
    let matrix_c_f32: Vec<f32> = vec![0.0; (M * N) as usize];

    // Element size depends on precision
    let element_size = if use_f16 { 2usize } else { 4usize };
    let num_elements_a = (M * K) as usize;
    let num_elements_b = (K * N) as usize;
    let num_elements_c = (M * N) as usize;

    // Create buffers
    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix A"),
        size: (num_elements_a * element_size) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix B"),
        size: (num_elements_b * element_size) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: (num_elements_c * element_size) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dimensions = Dimensions {
        m: M,
        n: N,
        k: K,
        stride: N,
    };
    let buffer_dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dimensions"),
        size: std::mem::size_of::<Dimensions>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (num_elements_c * element_size) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Upload data (convert to f16 if needed)
    if use_f16 {
        let matrix_a_f16: Vec<f16> = matrix_a_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let matrix_b_f16: Vec<f16> = matrix_b_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let matrix_c_f16: Vec<f16> = matrix_c_f32.iter().map(|&x| f16::from_f32(x)).collect();
        queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(&matrix_a_f16));
        queue.write_buffer(&buffer_b, 0, bytemuck::cast_slice(&matrix_b_f16));
        queue.write_buffer(&buffer_c, 0, bytemuck::cast_slice(&matrix_c_f16));
    } else {
        queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(&matrix_a_f32));
        queue.write_buffer(&buffer_b, 0, bytemuck::cast_slice(&matrix_b_f32));
        queue.write_buffer(&buffer_c, 0, bytemuck::cast_slice(&matrix_c_f32));
    }
    queue.write_buffer(&buffer_dims, 0, bytemuck::bytes_of(&dimensions));

    // Create bind group layout and bind group
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Cooperative Matrix Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Cooperative Matrix Bind Group"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_dims.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Cooperative Matrix Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Cooperative Matrix Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Cooperative Matrix Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cooperative Matrix Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Dispatch one workgroup per tile of the output
        compute_pass.dispatch_workgroups(M / config.m_size, N / config.m_size, 1);
    }

    // Copy result to staging buffer
    encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, staging_buffer.size());

    queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("Poll failed");
    receiver
        .recv_async()
        .await
        .expect("Channel receive failed")
        .expect("Buffer mapping failed");

    let data = buffer_slice.get_mapped_range();

    // Convert result back to f32 for comparison
    let result: Vec<f32> = if use_f16 {
        let result_f16: &[f16] = bytemuck::cast_slice(&data);
        result_f16.iter().map(|x| x.to_f32()).collect()
    } else {
        bytemuck::cast_slice::<_, f32>(&data).to_vec()
    };

    // Compute reference result on CPU for verification
    let mut reference = vec![0.0f32; (M * N) as usize];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += matrix_a_f32[(i * K + k) as usize] * matrix_b_f32[(k * N + j) as usize];
            }
            reference[(i * N + j) as usize] = sum;
        }
    }

    // Verify results (use larger tolerance for f16)
    let tolerance = if use_f16 { 0.1 } else { 0.01 };
    let mut max_error = 0.0f32;
    for i in 0..(M * N) as usize {
        let error = (result[i] - reference[i]).abs();
        max_error = max_error.max(error);
    }

    ExecuteResults {
        max_error,
        tolerance,
        matrix: result,
    }
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");
        crate::utils::add_web_nothing_to_see_msg();
        wasm_bindgen_futures::spawn_local(run());
    }
}

#[cfg(test)]
pub mod tests;
