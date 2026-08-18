//! Aggressive pure-wgpu reproducer for gfx-rs/wgpu#10085.
//!
//! Mirrors the fuzzgpu test-suite GPU pattern that crashes with heap
//! corruption (0xC0000374 / 0xC0000005) under >=3 concurrent dispatchers:
//! 8 threads on ONE shared Device/Queue, each iteration doing
//! queue.write_buffer uploads, a 2D matrix dispatch, a copy to a large
//! staging buffer, submit, map_async, poll, read, unmap, and per-iteration
//! bind-group churn — with buffers reused across iterations (pool-style).

use std::sync::{mpsc, Arc};
use std::time::Duration;

const N_THREADS: usize = 8;
const ITERS: usize = 40;
const MATRIX_DIM: u32 = 256;
const MATRIX_SIZE: u64 = (MATRIX_DIM * MATRIX_DIM) as u64 * 4;

const SHADER_SRC: &str = r#"
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * params.x + gid.x;
    out[idx] = idx + 1u;
}
"#;

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

struct Bufs {
    lens_a: wgpu::Buffer,
    lens_b: wgpu::Buffer,
    chars_a: wgpu::Buffer,
    chars_b: wgpu::Buffer,
    params: wgpu::Buffer,
    matrix: wgpu::Buffer,
    staging: wgpu::Buffer,
}

fn make_bufs(device: &wgpu::Device) -> Bufs {
    let mk_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    Bufs {
        lens_a: mk_buffer("lens_a", 4096, wgpu::BufferUsages::COPY_DST),
        lens_b: mk_buffer("lens_b", 4096, wgpu::BufferUsages::COPY_DST),
        chars_a: mk_buffer("chars_a", 64 * 1024, wgpu::BufferUsages::COPY_DST),
        chars_b: mk_buffer("chars_b", 64 * 1024, wgpu::BufferUsages::COPY_DST),
        params: mk_buffer(
            "params",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        ),
        matrix: mk_buffer(
            "matrix",
            MATRIX_SIZE,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ),
        staging: mk_buffer(
            "staging",
            MATRIX_SIZE,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        ),
    }
}

fn worker(ctx: Arc<Ctx>, id: usize) {
    let bufs = make_bufs(&ctx.device);
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matrix shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&ctx.bgl)],
        immediate_size: 0,
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    for i in 0..ITERS {
        let params_data: [u32; 4] = [MATRIX_DIM, MATRIX_DIM, 0, 0];

        ctx.queue.write_buffer(&bufs.lens_a, 0, bytemuck::cast_slice(&[0u32, MATRIX_DIM]));
        ctx.queue.write_buffer(&bufs.lens_b, 0, bytemuck::cast_slice(&[0u32, MATRIX_DIM]));
        ctx.queue.write_buffer(&bufs.chars_a, 0, &vec![0u8; 64 * 1024]);
        ctx.queue.write_buffer(&bufs.chars_b, 0, &vec![0u8; 64 * 1024]);
        ctx.queue.write_buffer(&bufs.params, 0, bytemuck::cast_slice(&params_data));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.matrix.as_entire_binding() },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(MATRIX_DIM / 16, MATRIX_DIM / 16, 1);
        }
        encoder.copy_buffer_to_buffer(&bufs.matrix, 0, &bufs.staging, 0, MATRIX_SIZE);
        ctx.queue.submit([encoder.finish()]);

        let slice = bufs.staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely());
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range().expect("mapped range");
                let raw: &[u32] = bytemuck::cast_slice(&data);
                debug_assert!(raw[0] == 1 && raw[MATRIX_DIM as usize] == MATRIX_DIM + 1);
                drop(data);
                bufs.staging.unmap();
            }
            other => panic!("readback failed: {:?}", other),
        }

        drop(bg);

        if i % 20 == 0 {
            println!("thread {id}: iter {i}");
        }
    }
    println!("thread {id}: done");
}

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matrix shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let ctx = Arc::new(Ctx {
        device,
        queue,
        pipeline,
        bgl,
    });

    let handles: Vec<_> = (0..N_THREADS)
        .map(|id| {
            let ctx = Arc::clone(&ctx);
            std::thread::spawn(move || worker(ctx, id))
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked");
    }
    println!("OK: {ITERS} iters x {N_THREADS} threads completed without crashing");
}
