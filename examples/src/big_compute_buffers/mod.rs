use std::{borrow::Cow, num::NonZeroU32, sync::Arc};
use wgpu::{util::DeviceExt, Features};

const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1;

pub async fn execute_gpu(numbers: &[f32]) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: Features::STORAGE_RESOURCE_BINDING_ARRAY |
                        // These features are required to use `binding_array` in your wgsl.
                    Features::BUFFER_BINDING_ARRAY |
                    Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,

                memory_hints: wgpu::MemoryHints::Performance,
                required_limits: wgpu::Limits {
                    max_buffer_size: MAX_BUFFER_SIZE,
                    ..Default::default()
                },
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, numbers).await
}

pub async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    numbers: &[f32],
) -> Option<Vec<f32>> {
    let (staging_buffers, storage_buffers, bind_group, compute_pipeline) = setup(device, numbers);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute pass descriptor"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, Some(&bind_group), &[]);

        cpass.dispatch_workgroups(MAX_DISPATCH_SIZE.min(numbers.len() as u32), 1, 1);
    }

    storage_buffers.iter().zip(staging_buffers.iter()).for_each(
        |(storage_buffer, staging_buffer)| {
            let stg_size = staging_buffer.size();

            encoder.copy_buffer_to_buffer(
                storage_buffer, // Source buffer
                0,
                staging_buffer, // Destination buffer
                0,
                stg_size,
            );
        },
    );

    queue.submit(Some(encoder.finish()));

    let mut buffer_slices = Vec::new();
    staging_buffers.iter().for_each(|sb| {
        buffer_slices.push(sb.slice(..));
    });

    let (sender, receiver) = flume::bounded(buffer_slices.len());
    let sender = Arc::new(sender);

    buffer_slices.iter().for_each(|bs| {
        let sender = sender.clone();
        bs.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        })
    });

    device.poll(wgpu::Maintain::wait());

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data: Vec<f32> = buffer_slices
            .iter()
            .flat_map(|bs| {
                let data = bs.get_mapped_range();
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data); // Drop to free buffer before unmap
                result
            })
            .collect();

        staging_buffers.iter().for_each(|sb| sb.unmap());

        Some(data)
    } else {
        log::error!("Failed to run compute on GPU!");
        None
    }
}

fn setup(
    device: &wgpu::Device,
    numbers: &[f32],
) -> (
    Vec<wgpu::Buffer>,
    Vec<wgpu::Buffer>,
    wgpu::BindGroup,
    wgpu::ComputePipeline,
) {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // Gets the size in bytes of the input.
    let input_size = std::mem::size_of_val(numbers) as wgpu::BufferAddress;
    let staging_buffers = create_staging_buffers(device, numbers);
    let storage_buffers = create_storage_buffers(device, numbers, input_size);

    let (bind_group_layout, bind_group) = setup_binds(&storage_buffers, device);

    let compute_pipeline = setup_pipeline(device, bind_group_layout, cs_module);
    (
        staging_buffers,
        storage_buffers,
        bind_group,
        compute_pipeline,
    )
}

fn setup_pipeline(
    device: &wgpu::Device,
    bind_group_layout: wgpu::BindGroupLayout,
    cs_module: wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn setup_binds(
    storage_buffers: &[wgpu::Buffer],
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let bind_group_entries: Vec<wgpu::BindGroupEntry> = storage_buffers
        .iter()
        .enumerate()
        .map(|(bind_idx, buffer)| wgpu::BindGroupEntry {
            binding: bind_idx as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();

    let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..storage_buffers.len())
        .map(|bind_idx| wgpu::BindGroupLayoutEntry {
            binding: bind_idx as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: Some(NonZeroU32::new(1).unwrap()),
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Custom Storage Bind Group Layout"),
        entries: &bind_group_layout_entries,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Combined Storage Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });
    (bind_group_layout, bind_group)
}

fn calculate_chunks(numbers: &[f32], max_buffer_size: u64) -> Vec<&[f32]> {
    let max_elements_per_chunk = max_buffer_size as usize / std::mem::size_of::<f32>(); // Calculate max f32 elements per buffer
    numbers.chunks(max_elements_per_chunk).collect()
}

fn create_storage_buffers(
    device: &wgpu::Device,
    numbers: &[f32],
    input_size: u64, // bytes..
) -> Vec<wgpu::Buffer> {
    if input_size > MAX_BUFFER_SIZE {
        let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

        chunks
            .iter()
            .enumerate()
            .map(|(e, seg)| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Storage Buffer-{}", e)),
                    contents: bytemuck::cast_slice(seg),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect()
    } else {
        vec![
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer-0"),
                contents: bytemuck::cast_slice(numbers),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        ]
    }
}

fn create_staging_buffers(device: &wgpu::Device, numbers: &[f32]) -> Vec<wgpu::Buffer> {
    let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

    (0..chunks.len())
        .map(|e| {
            let size = std::mem::size_of_val(chunks[e]) as u64;

            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("staging buffer-{}", e)),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect()
}

async fn run() {
    let numbers = {
        let bytes_per_gb = 1024 * 1024 * 1024;
        let bytes_per_f32 = std::mem::size_of::<f32>();
        let total_bytes = (1.0 * bytes_per_gb as f32) as usize;
        let elements = total_bytes / bytes_per_f32;

        vec![0.0; elements]
    };
    assert!(numbers.iter().all(|n| *n == 0.0));
    log::info!("All 0.0s");

    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await.unwrap();
    log::info!("GPU RUNTIME: {}ms", t1.elapsed().as_millis());

    assert_eq!(numbers.len(), results.len());
    assert!(results.iter().all(|n| *n == 1.0));
    log::info!("All 1.0s");
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
}

#[cfg(test)]
mod tests;
