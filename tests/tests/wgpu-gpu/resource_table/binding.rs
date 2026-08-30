//! Dirty-bit binding-state tests (work item 0.10).
//!
//! These exercise the conservative compute→compute memory-barrier scheme that
//! guards write→table-read (and table-read→write) hazards *inside* a table-bound
//! compute pass (see `plans/resource-table.md`, the *Barriers → Inside
//! table-bound compute passes* section). The scheme tracks two dirty bits and
//! emits one global memory barrier before a dependent dispatch.
//!
//! In M0 the table members themselves are read-only sampled textures (writing a
//! table-visible resource in a scope is a submit-time conflict error — work item
//! 0.9), so the writes here always target ordinary storage buffers *not* in the
//! table. The point of each test is therefore twofold: results stay correct, and
//! the emitted barriers are well-formed (the Vulkan validation-layer canary must
//! stay silent). The barrier is exercised whenever writable-binding dispatches
//! and table-reading dispatches interleave within one pass.

use wgpu::*;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer};

use super::common::{make_red_texture, read_u32s, table_params, texture_red, SAMPLING_SHADER};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_DIRTY_BITS_INTERLEAVED_WRITES_AND_TABLE_READS,
        RESOURCE_TABLE_DIRTY_BITS_BINDFUL_WRITE_BETWEEN_TABLE_READS,
        RESOURCE_TABLE_PIPELINE_SWITCH_CHANGES_SET_INDEX,
    ]);
}

/// The group-0 = {indices (read-only), output (read-write)} bind group layout
/// used by [`SAMPLING_SHADER`], with the table binding at set index 1.
fn sampling_bgl(ctx: &wgpu_test::TestingContext) -> BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sampling bgl"),
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
        })
}

/// A read-only storage-buffer layout entry at `binding` 0.
fn storage_ro_bgl(ctx: &wgpu_test::TestingContext, label: &str) -> BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
}

/// A read-write storage-buffer layout entry at `binding` 0.
fn storage_rw_bgl(ctx: &wgpu_test::TestingContext, label: &str) -> BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
}

// ---------------------------------------------------------------------------
// A: many interleaved (write + table-read) dispatches in one pass.
// ---------------------------------------------------------------------------

/// The plan's write→table-read interleaving stress test. One compute pass runs
/// eight dispatches; every dispatch both reads a distinct table slot and writes
/// its own storage buffer (`SAMPLING_SHADER` is `storage, read_write` on its
/// output). Every dispatch after the first therefore sees a prior write feeding
/// a table read and triggers the RAW dirty-bit barrier. All reads must come back
/// correct and the run must be validation-clean.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DIRTY_BITS_INTERLEAVED_WRITES_AND_TABLE_READS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const N: usize = 8;

            // One table with N distinct known-value members, uploaded on the
            // queue timeline.
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("stress table"),
                size: N as u32,
            });
            let mut _textures = Vec::with_capacity(N);
            for k in 0..N {
                let (texture, view) = make_red_texture(&ctx, texture_red(k));
                table.update(k as u32, &view).expect("bind member");
                _textures.push(texture);
            }

            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(SAMPLING_SHADER.into()),
            });
            let bgl = sampling_bgl(&ctx);
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            let make_buf = |usage: BufferUsages, init: Option<&[u8]>| {
                let buf = ctx.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 4,
                    usage,
                    mapped_at_creation: false,
                });
                if let Some(data) = init {
                    ctx.queue.write_buffer(&buf, 0, data);
                }
                buf
            };

            // Per-dispatch index buffer (which slot to read) + output + readback.
            let mut index_buffers = Vec::with_capacity(N);
            let mut outputs = Vec::with_capacity(N);
            let mut readbacks = Vec::with_capacity(N);
            let mut bind_groups = Vec::with_capacity(N);
            for k in 0..N {
                let idx = make_buf(
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    Some(&(k as u32).to_ne_bytes()),
                );
                let out = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
                let readback = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
                let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: out.as_entire_binding(),
                        },
                    ],
                });
                index_buffers.push(idx);
                outputs.push(out);
                readbacks.push(readback);
                bind_groups.push(bg);
            }

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("interleave pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_resource_table(Some(&table));
                for bg in &bind_groups {
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            for k in 0..N {
                encoder.copy_buffer_to_buffer(&outputs[k], 0, &readbacks[k], 0, 4);
            }
            ctx.queue.submit(Some(encoder.finish()));

            for (k, readback) in readbacks.iter().enumerate() {
                let got = read_u32s(&ctx, readback).await;
                assert_eq!(
                    got,
                    vec![texture_red(k) as u32],
                    "dispatch {k} read slot {k}"
                );
            }
        });

// ---------------------------------------------------------------------------
// B: a non-table bindful write dispatch between two table-reading dispatches.
// ---------------------------------------------------------------------------

/// Exercises *both* dirty-bit barrier directions in one pass. A table-reading
/// dispatch is followed by a dispatch that only writes a storage buffer (a
/// pipeline whose layout does not use the table), then by another table-reading
/// dispatch:
///
/// * the middle write-only dispatch, following the table read, trips the
///   `table-read → write` (WAR) barrier, and
/// * the final table read, following the write, trips the `write → table-read`
///   (RAW) barrier.
///
/// The table stays bound throughout (the non-table pipeline simply ignores it),
/// and all three outputs must be correct and validation-clean.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DIRTY_BITS_BINDFUL_WRITE_BETWEEN_TABLE_READS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED0: u8 = texture_red(0); // 10
            const RED1: u8 = texture_red(1); // 20

            let (_t0, v0) = make_red_texture(&ctx, RED0);
            let (_t1, v1) = make_red_texture(&ctx, RED1);
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("wr table"),
                size: 4,
            });
            table.update(0, &v0).expect("bind 0");
            table.update(1, &v1).expect("bind 1");

            let make_buf = |usage: BufferUsages, init: Option<&[u8]>| {
                let buf = ctx.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 4,
                    usage,
                    mapped_at_creation: false,
                });
                if let Some(data) = init {
                    ctx.queue.write_buffer(&buf, 0, data);
                }
                buf
            };

            // --- Table-reading pipeline (SAMPLING_SHADER, one group) ---------
            let sampling_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(SAMPLING_SHADER.into()),
            });
            // {read-only indices, read-write output} — shared by both pipelines'
            // single bind group.
            let rw_bgl = sampling_bgl(&ctx);
            let sampling_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&rw_bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let sampling_pipeline =
                ctx.device
                    .create_compute_pipeline(&ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&sampling_layout),
                        module: &sampling_module,
                        entry_point: Some("main"),
                        compilation_options: PipelineCompilationOptions::default(),
                        cache: None,
                    });

            // --- Write-only pipeline, no resource table ----------------------
            let write_shader = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] + 100u;
}
"#;
            let write_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(write_shader.into()),
            });
            // Reuses `rw_bgl`; this pipeline's layout simply does not use the table.
            let write_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&rw_bgl)],
                    immediate_size: 0,
                    uses_resource_table: false,
                });
            let write_pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&write_layout),
                    module: &write_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            // Buffers: two table-read outputs + one write-only in/out.
            let idx0 = make_buf(
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some(&0u32.to_ne_bytes()),
            );
            let idx1 = make_buf(
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some(&1u32.to_ne_bytes()),
            );
            let read_input = make_buf(
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some(&7u32.to_ne_bytes()),
            );
            let out_a = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let out_mid = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let out_c = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let rb_a = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
            let rb_mid = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
            let rb_c = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);

            let sampling_bg = |idx: &Buffer, out: &Buffer| {
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &rw_bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: out.as_entire_binding(),
                        },
                    ],
                })
            };
            let bg_a = sampling_bg(&idx0, &out_a);
            let bg_c = sampling_bg(&idx1, &out_c);
            let bg_mid = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &rw_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: read_input.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: out_mid.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("war/raw pass"),
                    timestamp_writes: None,
                });
                pass.set_resource_table(Some(&table));

                // Dispatch A: table read of slot 0 (+ bindful write of out_a).
                pass.set_pipeline(&sampling_pipeline);
                pass.set_bind_group(0, &bg_a, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                // Dispatch mid: write-only, no table (trips WAR barrier).
                pass.set_pipeline(&write_pipeline);
                pass.set_bind_group(0, &bg_mid, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                // Dispatch C: table read of slot 1 (trips RAW barrier). The table
                // is re-emitted here because set_pipeline dirtied it.
                pass.set_pipeline(&sampling_pipeline);
                pass.set_bind_group(0, &bg_c, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out_a, 0, &rb_a, 0, 4);
            encoder.copy_buffer_to_buffer(&out_mid, 0, &rb_mid, 0, 4);
            encoder.copy_buffer_to_buffer(&out_c, 0, &rb_c, 0, 4);
            ctx.queue.submit(Some(encoder.finish()));

            assert_eq!(
                read_u32s(&ctx, &rb_a).await,
                vec![RED0 as u32],
                "dispatch A read table slot 0"
            );
            assert_eq!(
                read_u32s(&ctx, &rb_mid).await,
                vec![107],
                "middle write-only dispatch"
            );
            assert_eq!(
                read_u32s(&ctx, &rb_c).await,
                vec![RED1 as u32],
                "dispatch C read table slot 1"
            );
        });

// ---------------------------------------------------------------------------
// C: switching pipelines that bind the table at different set indices.
// ---------------------------------------------------------------------------

/// The table must stay correctly bound when the pass switches between pipelines
/// whose layouts have different bind-group counts — the table binds at a
/// different set index for each (D15), so it must be re-emitted after every
/// switch. Also covers a redundant `set_resource_table` (same table again, a
/// no-op re-bind) and interleaving `set_bind_group` with the table binding.
///
/// Pipeline P1 has one bind group (table at set 1); pipeline P2 has two bind
/// groups (table at set 2). Both sample the shared table; all reads must be
/// correct and validation-clean.
#[apply(gpu_test!)]
static RESOURCE_TABLE_PIPELINE_SWITCH_CHANGES_SET_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(3); // 40

            let (_tex, view) = make_red_texture(&ctx, RED);
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("shared table"),
                size: 4,
            });
            table.update(0, &view).expect("bind slot 0");

            let make_buf = |usage: BufferUsages, init: Option<&[u8]>| {
                let buf = ctx.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 4,
                    usage,
                    mapped_at_creation: false,
                });
                if let Some(data) = init {
                    ctx.queue.write_buffer(&buf, 0, data);
                }
                buf
            };

            // --- P1: one group (indices + output), table at set 1 ------------
            let p1_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(SAMPLING_SHADER.into()),
            });
            let p1_bgl = sampling_bgl(&ctx);
            let p1_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&p1_bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let p1 = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&p1_layout),
                    module: &p1_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            // --- P2: two groups (indices@0, output@1), table at set 2 --------
            let p2_shader = r#"
enable resource_table;
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(1) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let tex = getResource<texture_2d<f32>>(indices[i]);
    output[i] = u32(round(textureLoad(tex, vec2<i32>(0, 0), 0).r * 255.0));
}
"#;
            let p2_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(p2_shader.into()),
            });
            let p2_idx_bgl = storage_ro_bgl(&ctx, "p2 idx bgl");
            let p2_out_bgl = storage_rw_bgl(&ctx, "p2 out bgl");
            let p2_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&p2_idx_bgl), Some(&p2_out_bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let p2 = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&p2_layout),
                    module: &p2_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            // Buffers for the three dispatches (P1, P2, P1 again).
            let idx = make_buf(
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some(&0u32.to_ne_bytes()),
            );
            let out0 = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let out1 = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let out2 = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let rb0 = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
            let rb1 = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
            let rb2 = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);

            let p1_bg = |out: &Buffer| {
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &p1_bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: idx.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: out.as_entire_binding(),
                        },
                    ],
                })
            };
            let p1_bg0 = p1_bg(&out0);
            let p1_bg2 = p1_bg(&out2);
            let p2_idx_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &p2_idx_bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: idx.as_entire_binding(),
                }],
            });
            let p2_out_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &p2_out_bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: out1.as_entire_binding(),
                }],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("set-index switch pass"),
                    timestamp_writes: None,
                });

                // Dispatch 0: P1, table at set 1.
                pass.set_pipeline(&p1);
                pass.set_bind_group(0, &p1_bg0, &[]);
                pass.set_resource_table(Some(&table));
                // Redundant re-bind of the same table: must be a harmless no-op.
                pass.set_resource_table(Some(&table));
                pass.dispatch_workgroups(1, 1, 1);

                // Dispatch 1: P2, table now at set 2 (different bind-group count).
                pass.set_pipeline(&p2);
                pass.set_bind_group(0, &p2_idx_bg, &[]);
                pass.set_bind_group(1, &p2_out_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);

                // Dispatch 2: back to P1, table returns to set 1.
                pass.set_pipeline(&p1);
                pass.set_bind_group(0, &p1_bg2, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out0, 0, &rb0, 0, 4);
            encoder.copy_buffer_to_buffer(&out1, 0, &rb1, 0, 4);
            encoder.copy_buffer_to_buffer(&out2, 0, &rb2, 0, 4);
            ctx.queue.submit(Some(encoder.finish()));

            assert_eq!(
                read_u32s(&ctx, &rb0).await,
                vec![RED as u32],
                "P1 (table at set 1) dispatch 0"
            );
            assert_eq!(
                read_u32s(&ctx, &rb1).await,
                vec![RED as u32],
                "P2 (table at set 2) dispatch 1"
            );
            assert_eq!(
                read_u32s(&ctx, &rb2).await,
                vec![RED as u32],
                "P1 (table back at set 1) dispatch 2"
            );
        });
