//! Reproducer for gfx-rs/wgpu#10085
//!
//! Heap corruption (`STATUS_HEAP_CORRUPTION` 0xC0000374, or `STATUS_ACCESS_VIOLATION`
//! 0xC0000005) when >=3 threads concurrently dispatch compute on a single shared
//! `wgpu::Device`, observed on Intel Iris Xe (Tiger Lake iGPU) on both DX12 and
//! Vulkan backends.
//!
//! # What this does
//!
//! Spawns N_THREADS threads, each submitting DISPATCHES_PER_THREAD compute dispatches
//! on the **same** shared `Device`/`Queue` pair (identical to the fuzzgpu test-suite
//! pattern that originally exposed the crash). Each dispatch creates a tiny storage
//! buffer, encodes a no-op compute shader, submits, polls, maps the result buffer,
//! reads it, and drops everything. This is the minimal loop that demonstrates the race.
//!
//! # Expected behaviour (fixed wgpu)
//! Exits 0 after all threads complete all dispatches.
//!
//! # Failure mode (unfixed wgpu)
//! Process aborts with exit code 0xC0000374 or 0xC0000005, typically within the
//! first 1–20 process runs when N_THREADS >= 3.
//!
//! # Running
//! ```
//! cargo run -p wgpu-bug-repro-10085 --release
//! ```
//! To reproduce the crash on unfixed wgpu, loop it:
//! ```powershell
//! for ($i = 1; $i -le 150; $i++) {
//!     cargo run -p wgpu-bug-repro-10085 --release
//!     if ($LASTEXITCODE -ne 0) { Write-Host "CRASH at $i"; break }
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

/// Number of concurrent dispatch threads. Must be >= 3 to trigger the race.
const N_THREADS: usize = 6;
/// Dispatches per thread per run.
const DISPATCHES_PER_THREAD: usize = 40;

// A minimal WGSL compute shader that writes 42u into results[0].
const SHADER_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@compute @workgroup_size(1)
fn main() {
    results[0] = 42u;
}
"#;

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuContext {
    async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;

        log::info!("Adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bug-10085"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .ok()?;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self { device, queue, pipeline, bgl })
    }

    /// One dispatch: create buffer → encode → submit → poll → map → verify → drop.
    fn dispatch_one(&self) {
        use wgpu::util::DeviceExt;
        let result_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: result_buf.as_entire_binding(),
            }],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result_buf, 0, &readback_buf, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();

        let data = slice.get_mapped_range().unwrap();
        let val: u32 = bytemuck::cast_slice(&data[..])[0];
        drop(data);
        readback_buf.unmap();
        assert_eq!(val, 42, "unexpected shader output");
    }
}

fn main() {
    env_logger::init();

    let ctx = pollster::block_on(GpuContext::new())
        .expect("No GPU adapter found — skipping bug-10085 repro");

    let ctx = Arc::new(ctx);

    let handles: Vec<_> = (0..N_THREADS)
        .map(|t| {
            let ctx = Arc::clone(&ctx);
            std::thread::spawn(move || {
                for i in 0..DISPATCHES_PER_THREAD {
                    ctx.dispatch_one();
                    log::trace!("thread {t} dispatch {i} ok");
                }
                log::info!("thread {t}: all {} dispatches ok", DISPATCHES_PER_THREAD);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    println!("All {} threads × {} dispatches completed without crash.", N_THREADS, DISPATCHES_PER_THREAD);
}
