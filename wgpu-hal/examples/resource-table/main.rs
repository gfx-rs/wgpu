//! Windowless wgpu-hal smoke test for the Vulkan resource-table (bindless)
//! implementation (work item 0.5 of `plans/resource-table.md`).
//!
//! It runs entirely on the real GPU with no window/surface:
//!
//! 1. Creates several small textures with distinct known texel values and
//!    transitions them to sampled usage.
//! 2. Creates a resource table larger than the number of textures, writes the
//!    texture views into the low slots, and deliberately leaves the high slots
//!    unwritten (exercising `PARTIALLY_BOUND`).
//! 3. Runs a compute shader that uses `enable resource_table;` +
//!    `getResource<texture_2d<f32>>(i)` with a *dynamic*, buffer-derived index
//!    (so the access is non-uniform), `textureLoad`s each texture, and writes
//!    the decoded red channel to a storage buffer.
//! 4. Reads the results back and asserts they exactly match the values written
//!    into the textures, permuted by the index buffer.
//!
//! Prints `PASS`, `SKIP` (if the adapter does not expose the feature), or
//! `FAIL`, and treats any Vulkan validation-layer error as a failure.

extern crate wgpu_hal as hal;

use hal::{Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _};

use std::{borrow::Cow, error::Error, iter, num::NonZeroU64, ptr};

type Api = hal::api::Vulkan;

/// Number of distinct textures we bind into the table.
const NUM_TEXTURES: usize = 4;
/// Table size; larger than `NUM_TEXTURES` so some slots stay unwritten.
const TABLE_SIZE: u32 = 8;
/// Permutation of texture indices fed to the shader, one per invocation. Using
/// a non-identity permutation proves the per-invocation dynamic indexing path.
const INDICES: [u32; NUM_TEXTURES] = [2, 0, 3, 1];

/// Red channel byte stored in texture `k`.
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

enum Outcome {
    Pass,
    Skip(String),
}

fn main() {
    env_logger::init();

    match run() {
        Ok(Outcome::Pass) => println!("resource-table smoke: PASS"),
        Ok(Outcome::Skip(reason)) => {
            println!("resource-table smoke: SKIP ({reason})");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("resource-table smoke: FAIL: {e}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<Outcome, Box<dyn Error>> {
    let instance_desc = hal::InstanceDescriptor {
        name: "resource-table-example",
        // Enables the Vulkan validation layers + debug messenger in debug
        // builds, exactly as the `halmark` example does.
        flags: wgpu_types::InstanceFlags::from_build_config().with_env(),
        memory_budget_thresholds: wgpu_types::MemoryBudgetThresholds::default(),
        backend_options: wgpu_types::BackendOptions::default(),
        telemetry: None,
        display: None,
    };
    let instance = unsafe { <Api as hal::Api>::Instance::init(&instance_desc)? };

    // Windowless: `enumerate_adapters(None)` (the surface hint is GLES-only).
    let (adapter, adapter_features) = {
        let mut adapters = unsafe { instance.enumerate_adapters(None) };
        if adapters.is_empty() {
            return Err("no Vulkan adapters found".into());
        }
        let exposed = adapters.swap_remove(0);
        (exposed.adapter, exposed.features)
    };

    let feature = wgpu_types::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE;
    if !adapter_features.contains(feature) {
        return Ok(Outcome::Skip(
            "adapter does not expose EXPERIMENTAL_SAMPLING_RESOURCE_TABLE".into(),
        ));
    }

    // M0 pipelines using tables require the unchecked add-on in addition to the
    // sampling bit (the checked path arrives with the M1 metadata buffer).
    let requested_features = wgpu_types::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
        | wgpu_types::Features::EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED;

    let hal::OpenDevice { device, queue } = unsafe {
        adapter.open(
            requested_features,
            &wgpu_types::Limits::default(),
            &wgpu_types::MemoryHints::default(),
        )?
    };

    let result = unsafe { run_inner(&device, &queue) };

    // Any validation-layer error recorded during the run is a failure. This is
    // only observable programmatically when built with the `validation_canary`
    // feature; without it, the debug messenger still logs errors, so run with
    // e.g. `RUST_LOG=wgpu_hal=warn` to surface them.
    #[cfg(feature = "validation_canary")]
    {
        let validation_errors = hal::VALIDATION_CANARY.get_and_reset();
        if !validation_errors.is_empty() {
            eprintln!(
                "resource-table smoke: FAIL: Vulkan validation layer reported {} error(s):\n{}",
                validation_errors.len(),
                validation_errors.join("\n")
            );
            // `process::exit` skips destructors, avoiding a confusing teardown
            // panic from resources leaked by an early error return above.
            std::process::exit(1);
        }
    }

    // On error, `run_inner` leaks the resources it created (the `?`-propagated
    // failure skips its teardown block); exit the process rather than let the
    // device's descriptor-allocator drop assert obscure the real error.
    if let Err(e) = result {
        eprintln!("resource-table smoke: FAIL: {e}");
        std::process::exit(1);
    }

    Ok(Outcome::Pass)
}

unsafe fn run_inner(
    device: &<Api as hal::Api>::Device,
    queue: &<Api as hal::Api>::Queue,
) -> Result<(), Box<dyn Error>> {
    // --- Shader ---------------------------------------------------------
    let module = naga::front::wgsl::Frontend::new().parse(SHADER)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        // The resource-table capability must be enabled for `getResource`.
        naga::valid::Capabilities::RESOURCE_TABLE,
    )
    .validate(&module)?;
    let shader = unsafe {
        device.create_shader_module(
            &hal::ShaderModuleDescriptor {
                label: Some("resource-table shader"),
                runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
            },
            hal::ShaderInput::Naga(hal::NagaShader {
                module: Cow::Owned(module),
                info,
                debug_source: None,
            }),
        )?
    };

    // --- Bind group layout / pipeline layout ----------------------------
    // One ordinary bind group: an index buffer and the output buffer.
    let bgl = unsafe {
        device.create_bind_group_layout(&hal::BindGroupLayoutDescriptor {
            label: Some("resource-table bgl"),
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[
                wgpu_types::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu_types::ShaderStages::COMPUTE,
                    ty: wgpu_types::BindingType::Buffer {
                        ty: wgpu_types::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu_types::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu_types::ShaderStages::COMPUTE,
                    ty: wgpu_types::BindingType::Buffer {
                        ty: wgpu_types::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })?
    };

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(&hal::PipelineLayoutDescriptor {
            label: Some("resource-table pipeline layout"),
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
            uses_resource_table: true,
        })?
    };

    let constants = naga::back::PipelineConstants::default();
    let pipeline = unsafe {
        device.create_compute_pipeline(&hal::ComputePipelineDescriptor {
            label: Some("resource-table pipeline"),
            layout: &pipeline_layout,
            stage: hal::ProgrammableStage {
                module: &shader,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        })?
    };

    // --- Textures + staging uploads ------------------------------------
    let texture_desc = hal::TextureDescriptor {
        label: Some("resource-table texture"),
        size: wgpu_types::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu_types::TextureDimension::D2,
        format: wgpu_types::TextureFormat::Rgba8Unorm,
        usage: wgpu_types::TextureUses::COPY_DST | wgpu_types::TextureUses::RESOURCE,
        memory_flags: hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    let mut textures = Vec::with_capacity(NUM_TEXTURES);
    let mut texture_views = Vec::with_capacity(NUM_TEXTURES);
    let mut staging_buffers = Vec::with_capacity(NUM_TEXTURES);

    for k in 0..NUM_TEXTURES {
        let texture = unsafe { device.create_texture(&texture_desc)? };

        let texel: [u8; 4] = [texture_red(k), 0, 0, 255];
        let staging = unsafe {
            device.create_buffer(&hal::BufferDescriptor {
                label: Some("resource-table staging"),
                size: texel.len() as wgpu_types::BufferAddress,
                usage: wgpu_types::BufferUses::MAP_WRITE | wgpu_types::BufferUses::COPY_SRC,
                memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
            })?
        };
        unsafe {
            let mapping = device.map_buffer(&staging, 0..texel.len() as u64)?;
            ptr::copy_nonoverlapping(texel.as_ptr(), mapping.ptr.as_ptr(), texel.len());
            device.unmap_buffer(&staging);
            assert!(mapping.is_coherent);
        }

        let view = unsafe {
            device.create_texture_view(
                &texture,
                &hal::TextureViewDescriptor {
                    label: Some("resource-table view"),
                    format: texture_desc.format,
                    dimension: wgpu_types::TextureViewDimension::D2,
                    usage: wgpu_types::TextureUses::RESOURCE,
                    range: wgpu_types::ImageSubresourceRange::default(),
                },
            )?
        };

        textures.push(texture);
        texture_views.push(view);
        staging_buffers.push(staging);
    }

    // --- Resource table -------------------------------------------------
    let table = unsafe {
        device.create_resource_table(&hal::ResourceTableDescriptor {
            label: Some("resource-table"),
            size: TABLE_SIZE,
        })?
    };
    // Populate the low slots; slots `NUM_TEXTURES..TABLE_SIZE` intentionally
    // stay unwritten (the shader never reads them; PARTIALLY_BOUND allows this).
    for (slot, view) in texture_views.iter().enumerate() {
        unsafe {
            device.update_table_slot(
                &table,
                slot as u32,
                hal::ResourceTableUpdate::SampledTextureView(view),
            );
        }
    }

    // --- Index / output / readback buffers ------------------------------
    let index_bytes: Vec<u8> = INDICES.iter().flat_map(|i| i.to_ne_bytes()).collect();
    let buffer_size = index_bytes.len() as wgpu_types::BufferAddress;

    let index_buffer = unsafe {
        let buffer = device.create_buffer(&hal::BufferDescriptor {
            label: Some("resource-table indices"),
            size: buffer_size,
            usage: wgpu_types::BufferUses::MAP_WRITE | wgpu_types::BufferUses::STORAGE_READ_ONLY,
            memory_flags: hal::MemoryFlags::PREFER_COHERENT,
        })?;
        let mapping = device.map_buffer(&buffer, 0..buffer_size)?;
        ptr::copy_nonoverlapping(
            index_bytes.as_ptr(),
            mapping.ptr.as_ptr(),
            index_bytes.len(),
        );
        device.unmap_buffer(&buffer);
        assert!(mapping.is_coherent);
        buffer
    };

    let output_buffer = unsafe {
        device.create_buffer(&hal::BufferDescriptor {
            label: Some("resource-table output"),
            size: buffer_size,
            usage: wgpu_types::BufferUses::STORAGE_READ_WRITE | wgpu_types::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::empty(),
        })?
    };

    let readback_buffer = unsafe {
        device.create_buffer(&hal::BufferDescriptor {
            label: Some("resource-table readback"),
            size: buffer_size,
            usage: wgpu_types::BufferUses::MAP_READ | wgpu_types::BufferUses::COPY_DST,
            memory_flags: hal::MemoryFlags::PREFER_COHERENT,
        })?
    };

    let bind_group = unsafe {
        let index_binding =
            hal::BufferBinding::new_unchecked(&index_buffer, 0, NonZeroU64::new(buffer_size));
        let output_binding =
            hal::BufferBinding::new_unchecked(&output_buffer, 0, NonZeroU64::new(buffer_size));
        device.create_bind_group(&hal::BindGroupDescriptor {
            label: Some("resource-table bind group"),
            layout: &bgl,
            buffers: &[index_binding, output_binding],
            samplers: &[],
            textures: &[],
            acceleration_structures: &[],
            external_textures: &[],
            entries: &[
                hal::BindGroupEntry {
                    binding: 0,
                    resource_index: 0,
                    count: 1,
                },
                hal::BindGroupEntry {
                    binding: 1,
                    resource_index: 1,
                    count: 1,
                },
            ],
        })?
    };

    // --- Record + submit ------------------------------------------------
    let mut encoder = unsafe {
        device.create_command_encoder(&hal::CommandEncoderDescriptor {
            label: Some("resource-table encoder"),
            queue,
        })?
    };
    unsafe { encoder.begin_encoding(Some("resource-table"))? };

    // Upload every texture and transition it to sampled usage.
    for (texture, staging) in textures.iter().zip(staging_buffers.iter()) {
        unsafe {
            encoder.transition_buffers(iter::once(hal::BufferBarrier {
                buffer: staging,
                usage: hal::StateTransition {
                    from: wgpu_types::BufferUses::empty(),
                    to: wgpu_types::BufferUses::COPY_SRC,
                },
            }));
            encoder.transition_textures(iter::once(hal::TextureBarrier {
                texture,
                range: wgpu_types::ImageSubresourceRange::default(),
                usage: hal::StateTransition {
                    from: wgpu_types::TextureUses::UNINITIALIZED,
                    to: wgpu_types::TextureUses::COPY_DST,
                },
                queue_family_ownership_transfer: None,
            }));
            encoder.copy_buffer_to_texture(
                staging,
                texture,
                iter::once(hal::BufferTextureCopy {
                    buffer_layout: wgpu_types::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: None,
                    },
                    texture_base: hal::TextureCopyBase {
                        origin: wgpu_types::Origin3d::ZERO,
                        mip_level: 0,
                        array_layer: 0,
                        aspect: hal::FormatAspects::COLOR,
                    },
                    size: hal::CopyExtent {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                }),
            );
            encoder.transition_textures(iter::once(hal::TextureBarrier {
                texture,
                range: wgpu_types::ImageSubresourceRange::default(),
                usage: hal::StateTransition {
                    from: wgpu_types::TextureUses::COPY_DST,
                    to: wgpu_types::TextureUses::RESOURCE,
                },
                queue_family_ownership_transfer: None,
            }));
        }
    }

    // Make the index/output buffers available to the compute pass.
    unsafe {
        encoder.transition_buffers(
            [
                hal::BufferBarrier {
                    buffer: &index_buffer,
                    usage: hal::StateTransition {
                        from: wgpu_types::BufferUses::empty(),
                        to: wgpu_types::BufferUses::STORAGE_READ_ONLY,
                    },
                },
                hal::BufferBarrier {
                    buffer: &output_buffer,
                    usage: hal::StateTransition {
                        from: wgpu_types::BufferUses::empty(),
                        to: wgpu_types::BufferUses::STORAGE_READ_WRITE,
                    },
                },
            ]
            .into_iter(),
        );
    }

    unsafe {
        encoder.begin_compute_pass(&hal::ComputePassDescriptor {
            label: Some("resource-table pass"),
            timestamp_writes: None,
        });
        encoder.set_compute_pipeline(&pipeline);
        encoder.set_bind_group(&pipeline_layout, 0, &bind_group, &[]);
        // The table binds at set index == the layout's bind-group count (1).
        encoder.set_resource_table(&pipeline_layout, 1, &table);
        encoder.dispatch_workgroups([1, 1, 1]);
        encoder.end_compute_pass();
    }

    // Copy the results into a mappable buffer.
    unsafe {
        encoder.transition_buffers(iter::once(hal::BufferBarrier {
            buffer: &output_buffer,
            usage: hal::StateTransition {
                from: wgpu_types::BufferUses::STORAGE_READ_WRITE,
                to: wgpu_types::BufferUses::COPY_SRC,
            },
        }));
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            &readback_buffer,
            iter::once(hal::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: wgpu_types::BufferSize::new(buffer_size).unwrap(),
            }),
        );
    }

    let fence_value = 1;
    let mut fence = unsafe { device.create_fence()? };
    unsafe {
        let cmd_buf = encoder.end_encoding()?;
        queue.submit(&[&cmd_buf], &[], (&mut fence, fence_value))?;
        device.wait(&fence, fence_value, None)?;
        encoder.reset_all(iter::once(cmd_buf));
    }

    // --- Read back + verify --------------------------------------------
    let mut results = [0u32; NUM_TEXTURES];
    unsafe {
        let mapping = device.map_buffer(&readback_buffer, 0..buffer_size)?;
        ptr::copy_nonoverlapping(
            mapping.ptr.as_ptr(),
            results.as_mut_ptr() as *mut u8,
            index_bytes.len(),
        );
        device.unmap_buffer(&readback_buffer);
        assert!(mapping.is_coherent);
    }

    let expected: Vec<u32> = INDICES
        .iter()
        .map(|&idx| texture_red(idx as usize) as u32)
        .collect();

    println!("resource-table smoke: indices={INDICES:?} expected={expected:?} got={results:?}");

    // --- Cleanup --------------------------------------------------------
    unsafe {
        device.destroy_bind_group(bind_group);
        device.destroy_resource_table(table);
        device.destroy_buffer(readback_buffer);
        device.destroy_buffer(output_buffer);
        device.destroy_buffer(index_buffer);
        for staging in staging_buffers {
            device.destroy_buffer(staging);
        }
        for view in texture_views {
            device.destroy_texture_view(view);
        }
        for texture in textures {
            device.destroy_texture(texture);
        }
        device.destroy_compute_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);
        device.destroy_bind_group_layout(bgl);
        device.destroy_shader_module(shader);
        device.destroy_fence(fence);
        drop(encoder);
    }

    if results.as_slice() != expected.as_slice() {
        return Err(format!("mismatch: expected {expected:?}, got {results:?}").into());
    }

    Ok(())
}
