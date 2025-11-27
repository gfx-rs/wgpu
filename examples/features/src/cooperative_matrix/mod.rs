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
//! experimental. The 8x8 f32 matrix format is well supported on Metal
//! (simdgroup matrix operations). On Vulkan, support depends on hardware -
//! most GPUs support f16 at 16x16 sizes, so 8x8 f32 support may be limited.

use bytemuck::{Pod, Zeroable};

/// Matrix dimensions for our example
const M: u32 = 64; // Rows of A and C
const N: u32 = 64; // Cols of B and C
const K: u32 = 64; // Cols of A, Rows of B
const TILE_SIZE: u32 = 8;

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

    // Check if cooperative matrix is supported
    let adapter_features = adapter.features();
    if !adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        log::error!(
            "Cooperative matrix is not supported on this adapter.\n\
            This feature requires:\n\
            - Metal: Apple7+ (A14/M1) with MSL 2.3+\n\
            - Vulkan: VK_KHR_cooperative_matrix with 8x8 f32 support (rare)\n\
            Most Vulkan GPUs (NVIDIA, AMD) only support f16 inputs at 16x16 sizes."
        );
        return;
    }

    // Request device with experimental features enabled
    let (device, queue) = unsafe {
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Cooperative Matrix Device"),
                required_features: wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::enabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device")
    };

    // Create the shader module using the standard validated path
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cooperative Matrix Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    // Initialize matrices
    // A is MxK, B is KxN, C is MxN (result)
    let matrix_a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let matrix_b: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
    let matrix_c: Vec<f32> = vec![0.0; (M * N) as usize];

    // Create buffers
    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix A"),
        size: (matrix_a.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix B"),
        size: (matrix_b.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: (matrix_c.len() * std::mem::size_of::<f32>()) as u64,
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
        size: (matrix_c.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Upload data
    queue.write_buffer(&buffer_a, 0, bytemuck::cast_slice(&matrix_a));
    queue.write_buffer(&buffer_b, 0, bytemuck::cast_slice(&matrix_b));
    queue.write_buffer(&buffer_c, 0, bytemuck::cast_slice(&matrix_c));
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
        bind_group_layouts: &[&bind_group_layout],
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
        // Dispatch one workgroup per 8x8 tile of the output
        compute_pass.dispatch_workgroups(M / TILE_SIZE, N / TILE_SIZE, 1);
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
    let result: &[f32] = bytemuck::cast_slice(&data);

    // Compute reference result on CPU for verification
    let mut reference = vec![0.0f32; (M * N) as usize];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += matrix_a[(i * K + k) as usize] * matrix_b[(k * N + j) as usize];
            }
            reference[(i * N + j) as usize] = sum;
        }
    }

    // Verify results
    let mut max_error = 0.0f32;
    for i in 0..(M * N) as usize {
        let error = (result[i] - reference[i]).abs();
        max_error = max_error.max(error);
    }

    log::info!("Matrix multiplication {M}x{K}x{N} completed!");
    log::info!("Max error vs CPU reference: {max_error:.6}");

    if max_error < 0.01 {
        log::info!("✓ Results match CPU reference within tolerance");
    } else {
        log::warn!("✗ Results differ from CPU reference");
    }

    // Print a small sample of the result
    log::info!("Sample of result matrix C (top-left 4x4):");
    for i in 0..4 {
        let row: Vec<String> = (0..4)
            .map(|j| format!("{:6.2}", result[i * N as usize + j]))
            .collect();
        log::info!("  [{}]", row.join(", "));
    }

    drop(data);
    staging_buffer.unmap();
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
