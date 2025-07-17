//! This example shows you a potential course for when your 'data' is too large
//! for a single Buffer.
//!
//! A lot of things aren't explained here via comments. See hello-compute and
//! repeated-compute for code that is more thoroughly commented.

use std::num::NonZeroU32;
use wgpu::{util::DeviceExt, Features};

// These are set by the minimum required defaults for webgpu.
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1;

pub async fn execute_gpu(numbers: &[f32]) -> Vec<f32> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            // These features are required to use `binding_array` in your wgsl.
            // Without them your shader may fail to compile.
            required_features: Features::BUFFER_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
                | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            memory_hints: wgpu::MemoryHints::Performance,
            required_limits: wgpu::Limits {
                max_buffer_size: MAX_BUFFER_SIZE,
                max_binding_array_elements_per_shader_stage: 8,
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, numbers).await
}

pub async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    numbers: &[f32],
) -> Vec<f32> {
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

    for (storage_buffer, staging_buffer) in storage_buffers.iter().zip(staging_buffers.iter()) {
        let stg_size = staging_buffer.size();

        encoder.copy_buffer_to_buffer(
            storage_buffer, // Source buffer
            0,
            staging_buffer, // Destination buffer
            0,
            stg_size,
        );
    }

    queue.submit(Some(encoder.finish()));

    for staging_buffer in &staging_buffers {
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
    }

    device.poll(wgpu::PollType::Wait).unwrap();

    let mut data = Vec::new();
    for staging_buffer in &staging_buffers {
        let slice = staging_buffer.slice(..);
        let mapped = slice.get_mapped_range();
        data.extend_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        staging_buffer.unmap();
    }

    data
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
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let staging_buffers = create_staging_buffers(device, numbers);
    let storage_buffers = create_storage_buffers(device, numbers);

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
    let max_elements_per_chunk = max_buffer_size as usize / std::mem::size_of::<f32>();
    numbers.chunks(max_elements_per_chunk).collect()
}

fn create_storage_buffers(device: &wgpu::Device, numbers: &[f32]) -> Vec<wgpu::Buffer> {
    let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

    chunks
        .iter()
        .enumerate()
        .map(|(e, seg)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Storage Buffer-{e}")),
                contents: bytemuck::cast_slice(seg),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect()
}

fn create_staging_buffers(device: &wgpu::Device, numbers: &[f32]) -> Vec<wgpu::Buffer> {
    let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

    (0..chunks.len())
        .map(|e| {
            let size = std::mem::size_of_val(chunks[e]) as u64;

            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("staging buffer-{e}")),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", allow(clippy::allow_attributes, dead_code))]
async fn run() {
    let numbers = {
        const BYTES_PER_GB: usize = 1024 * 1024 * 1024;
        // 4 bytes per f32
        let elements = (BYTES_PER_GB as f32 / 4.0) as usize;
        vec![0.0; elements]
    };
    assert!(numbers.iter().all(|n| *n == 0.0));
    log::info!("All 0.0s");
    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await;
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
#[cfg(not(target_arch = "wasm32"))]
mod tests;
