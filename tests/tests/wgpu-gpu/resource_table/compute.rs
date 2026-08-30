//! Compute-shader smoke test for the resource table, end to end through the
//! public `wgpu` API.
//!
//! Mirrors the wgpu-hal windowless smoke example (work item 0.5), but drives the
//! whole stack via `wgpu`: `Device::create_resource_table`, the public
//! `ResourceTable::update` / `insert_binding` slot-update API, and
//! `ComputePass::set_resource_table`.
//!
//! Flow:
//! 1. Create several 1x1 textures with distinct known red values, uploaded with
//!    `queue.write_texture`. The uploads ride the queue timeline
//!    (`pending_writes`), i.e. a *different* command buffer from the sampling
//!    pass — same-command-buffer write-then-table-sample is a documented M0
//!    limitation, so the textures must be established outside the sampling CB.
//! 2. Create a table larger than the texture count; bind the textures into the
//!    low slots (using both `update` by explicit slot and `insert_binding`), and
//!    leave the high slots unwritten (exercising sparse / partially-bound tables).
//! 3. Run a compute shader that reads `getResource<texture_2d<f32>>(slot)` with a
//!    per-invocation, buffer-derived slot index (non-uniform), `textureLoad`s the
//!    texel, and writes the decoded red channel to a storage buffer.
//! 4. Read the storage buffer back and assert the values match the textures,
//!    permuted by the index buffer.

use wgpu::*;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

/// Number of distinct textures bound into the table.
const NUM_TEXTURES: usize = 4;
/// Table size; larger than `NUM_TEXTURES` so some slots stay unwritten.
const TABLE_SIZE: u32 = 8;
/// Permutation of slot indices fed to the shader, one per invocation. A
/// non-identity permutation proves the per-invocation dynamic indexing path.
const INDICES: [u32; NUM_TEXTURES] = [2, 0, 3, 1];

/// Red channel byte stored in (and expected back from) texture `k`.
fn texture_red(k: usize) -> u8 {
    ((k + 1) * 10) as u8
}

const SHADER: &str = r#"
enable resource_table;

@group(0) @binding(0)
var<storage, read> indices: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let slot = indices[i];
    let tex = getResource<texture_2d<f32>>(slot);
    let texel = textureLoad(tex, vec2<i32>(0, 0), 0);
    output[i] = u32(round(texel.r * 255.0));
}
"#;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(RESOURCE_TABLE_COMPUTE_SMOKE);
}

#[apply(gpu_test!)]
static RESOURCE_TABLE_COMPUTE_SMOKE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
                    | Features::EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED,
            )
            // The default harness limits are WebGL2-downlevel, which forbid
            // storage buffers in the compute stage; this test needs two.
            .limits(Limits::downlevel_defaults())
            // M0 is Vulkan-only; other backends have no table implementation yet.
            .skip(FailureCase::backend(!Backends::VULKAN)),
    )
    .run_async(|ctx| async move { resource_table_compute_smoke(ctx).await });

async fn resource_table_compute_smoke(ctx: TestingContext) {
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("resource-table compute shader"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });

    // --- Textures, uploaded on the queue timeline (outside the sampling CB) ---
    let mut views = Vec::with_capacity(NUM_TEXTURES);
    for k in 0..NUM_TEXTURES {
        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("resource-table texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texel: [u8; 4] = [texture_red(k), 0, 0, 255];
        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &texel,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        views.push(texture.create_view(&TextureViewDescriptor::default()));
    }

    // --- Resource table -------------------------------------------------
    let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
        label: Some("resource-table"),
        size: TABLE_SIZE,
    });
    assert_eq!(table.size(), TABLE_SIZE);

    // Bind textures 0/1 by explicit slot, and 2/3 via `insert_binding` (which
    // must pick the lowest currently-empty slot: 2, then 3). Slots
    // `NUM_TEXTURES..TABLE_SIZE` stay unwritten.
    table.update(0, &views[0]).expect("update slot 0");
    table.update(1, &views[1]).expect("update slot 1");
    assert_eq!(
        table.insert_binding(&views[2]).expect("insert view 2"),
        2,
        "insert_binding must return the lowest empty slot"
    );
    assert_eq!(
        table.insert_binding(&views[3]).expect("insert view 3"),
        3,
        "insert_binding must return the lowest empty slot"
    );

    // --- Index / output / readback buffers ------------------------------
    let index_bytes: Vec<u8> = INDICES.iter().flat_map(|i| i.to_ne_bytes()).collect();
    let buffer_size = index_bytes.len() as BufferAddress;

    let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("resource-table indices"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&index_buffer, 0, &index_bytes);

    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("resource-table output"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("resource-table readback"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Bind group / pipeline layout / pipeline ------------------------
    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("resource-table bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("resource-table bind group"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: index_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("resource-table pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
            uses_resource_table: true,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("resource-table pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

    // --- Record + submit ------------------------------------------------
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("resource-table pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_resource_table(Some(&table));
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, buffer_size);
    ctx.queue.submit(Some(encoder.finish()));

    // --- Read back + verify --------------------------------------------
    readback_buffer.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

    let data = readback_buffer.slice(..).get_mapped_range().unwrap();
    let got: &[u32] = bytemuck::cast_slice(&data);

    let expected: Vec<u32> = INDICES
        .iter()
        .map(|&idx| texture_red(idx as usize) as u32)
        .collect();

    assert_eq!(
        got,
        expected.as_slice(),
        "indices={INDICES:?} expected={expected:?} got={got:?}"
    );
}
