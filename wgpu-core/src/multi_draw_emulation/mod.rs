use crate::{
    command::RenderPassErrorInner,
    device::{Device, DeviceError},
    hal_label,
    lock::{rank, Mutex},
    pipeline::{CreateComputePipelineError, CreateShaderModuleError},
    resource::{Buffer, InvalidResourceError, Labeled, RawResourceAccess},
    snatch::SnatchGuard,
};
use alloc::{
    borrow::Cow,
    boxed::Box,
    string::ToString,
    sync::Arc,
    vec::Vec,
};
use core::num::NonZeroU64;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
enum CreateMultiDrawEmulationPipelineError {
    #[error(transparent)]
    DeviceError(#[from] DeviceError),
    #[error(transparent)]
    ShaderModule(#[from] CreateShaderModuleError),
    #[error(transparent)]
    ComputePipeline(#[from] CreateComputePipelineError),
}

struct MultiDrawEmulationInner {
    module: Box<dyn hal::DynShaderModule>,
    bgl: Box<dyn hal::DynBindGroupLayout>,
    pipeline_layout: Box<dyn hal::DynPipelineLayout>,
    pipeline: Box<dyn hal::DynComputePipeline>,
    temp_pool: Mutex<Vec<TempPoolEntry>>,
}

/// This machinery requires the following limits:
///
/// - max_bind_groups: 1,
/// - max_storage_buffers_per_shader_stage: 3,
/// - max_immediate_size: 16,
///
/// These are all indirectly satisfied by `DownlevelFlags::INDIRECT_EXECUTION`, which is also
/// required for this module's functionality to work.
pub(crate) struct MultiDrawEmulation {
    inner: Option<MultiDrawEmulationInner>,
}

struct TempPoolEntry {
    buffer: Box<dyn hal::DynBuffer>,
    size: u64,
}

pub(crate) struct PendingDraw {
    pub(crate) temp_buffer_index: usize,
    pub(crate) src_buffer: Arc<Buffer>,
    pub(crate) count_buffer: Arc<Buffer>,
    pub(crate) src_offset: u64,
    pub(crate) count_offset: u64,
    pub(crate) max_count: u32,
    pub(crate) stride_u32: u32,
}

pub(crate) struct MultiDrawResources {
    device: Arc<Device>,
    temp_entries: Vec<TempPoolEntry>,
    bind_groups: Vec<Box<dyn hal::DynBindGroup>>,
}

impl MultiDrawEmulation {
    pub(crate) fn new(
        device: &dyn hal::DynDevice,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<Self, DeviceError> {
        let inner = match MultiDrawEmulationInner::new(device, instance_flags) {
            Ok(inner) => inner,
            Err(e) => {
                log::error!("multi-draw-emulation error: {e:?}");
                return Err(DeviceError::Lost);
            }
        };
        Ok(Self {
            inner: Some(inner),
        })
    }

    pub(crate) fn dispose(self, device: &dyn hal::DynDevice) {
        let inner = self.inner.unwrap();
        for entry in inner.temp_pool.into_inner() {
            unsafe { device.destroy_buffer(entry.buffer) };
        }
        unsafe { device.destroy_compute_pipeline(inner.pipeline) };
        unsafe { device.destroy_pipeline_layout(inner.pipeline_layout) };
        unsafe { device.destroy_bind_group_layout(inner.bgl) };
        unsafe { device.destroy_shader_module(inner.module) };
    }

    fn acquire_temp_entry(
        &self,
        device: &dyn hal::DynDevice,
        size: u64,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<TempPoolEntry, DeviceError> {
        let inner = self.inner.as_ref().unwrap();
        let mut pool = inner.temp_pool.lock();

        if let Some(idx) = pool.iter().position(|e| e.size >= size) {
            return Ok(pool.swap_remove(idx));
        }

        let buffer_desc = hal::BufferDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw emulation temp buffer"),
                instance_flags,
            ),
            size,
            usage: wgt::BufferUses::STORAGE_READ_WRITE | wgt::BufferUses::INDIRECT,
            memory_flags: hal::MemoryFlags::empty(),
        };
        let buffer =
            unsafe { device.create_buffer(&buffer_desc) }.map_err(DeviceError::from_hal)?;

        Ok(TempPoolEntry { buffer, size })
    }

    fn release_temp_entries(&self, entries: impl Iterator<Item = TempPoolEntry>) {
        let inner = self.inner.as_ref().unwrap();
        inner.temp_pool.lock().extend(entries);
    }

    pub(crate) fn inject_emulation_pass(
        &self,
        device: &Arc<Device>,
        resources: &mut MultiDrawResources,
        encoder: &mut dyn hal::DynCommandEncoder,
        pending: Vec<PendingDraw>,
        snatch_guard: &SnatchGuard,
    ) -> Result<(), RenderPassErrorInner> {
        if pending.is_empty() {
            return Ok(());
        }

        let inner = self.inner.as_ref().unwrap();

        {
            let mut barriers: Vec<hal::BufferBarrier<dyn hal::DynBuffer>> = Vec::new();

            for p in &pending {
                let src_buf = p.src_buffer.try_raw(snatch_guard).map_err(|_| {
                    RenderPassErrorInner::InvalidResource(InvalidResourceError(
                        p.src_buffer.error_ident(),
                    ))
                })?;
                let count_buf = p.count_buffer.try_raw(snatch_guard).map_err(|_| {
                    RenderPassErrorInner::InvalidResource(InvalidResourceError(
                        p.count_buffer.error_ident(),
                    ))
                })?;
                let temp_buffer = resources.get_temp_buffer(p.temp_buffer_index);
                barriers.push(hal::BufferBarrier {
                    buffer: src_buf,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::INDIRECT,
                        to: wgt::BufferUses::STORAGE_READ_ONLY,
                    },
                });
                barriers.push(hal::BufferBarrier {
                    buffer: count_buf,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::INDIRECT,
                        to: wgt::BufferUses::STORAGE_READ_ONLY,
                    },
                });
                barriers.push(hal::BufferBarrier {
                    buffer: temp_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::INDIRECT,
                        to: wgt::BufferUses::STORAGE_READ_WRITE,
                    },
                });
            }

            unsafe { encoder.transition_buffers(&barriers) };
        }

        let compute_desc = hal::ComputePassDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw indirect count emulation pass"),
                device.instance_flags,
            ),
            timestamp_writes: None,
        };
        unsafe { encoder.begin_compute_pass(&compute_desc) };
        unsafe { encoder.set_compute_pipeline(inner.pipeline.as_ref()) };

        let alignment = device.limits.min_storage_buffer_offset_alignment as u64;

        for p in &pending {
            let temp_buffer = resources.get_temp_buffer(p.temp_buffer_index);
            let src_buf = p.src_buffer.try_raw(snatch_guard).map_err(|_| {
                RenderPassErrorInner::InvalidResource(InvalidResourceError(
                    p.src_buffer.error_ident(),
                ))
            })?;
            let count_buf = p.count_buffer.try_raw(snatch_guard).map_err(|_| {
                RenderPassErrorInner::InvalidResource(InvalidResourceError(
                    p.count_buffer.error_ident(),
                ))
            })?;

            let aligned_src_offset = p.src_offset - p.src_offset % alignment;
            let src_remainder_u32 = ((p.src_offset - aligned_src_offset) / 4) as u32;

            let aligned_count_offset = p.count_offset - p.count_offset % alignment;
            let count_remainder_u32 = ((p.count_offset - aligned_count_offset) / 4) as u32;
            let count_binding_size =
                NonZeroU64::new((count_remainder_u32 as u64 + 1) * 4).unwrap();

            let temp_size =
                NonZeroU64::new(p.max_count as u64 * p.stride_u32 as u64 * 4).unwrap();

            let bg_desc = hal::BindGroupDescriptor {
                label: hal_label(
                    Some("(wgpu internal) Multi-draw emulation bind group"),
                    device.instance_flags,
                ),
                layout: inner.bgl.as_ref(),
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
                    hal::BindGroupEntry {
                        binding: 2,
                        resource_index: 2,
                        count: 1,
                    },
                ],
                buffers: &[
                    hal::BufferBinding::new_unchecked(src_buf, aligned_src_offset, None),
                    hal::BufferBinding::new_unchecked(temp_buffer, 0, Some(temp_size)),
                    hal::BufferBinding::new_unchecked(
                        count_buf,
                        aligned_count_offset,
                        Some(count_binding_size),
                    ),
                ],
                samplers: &[],
                textures: &[],
                acceleration_structures: &[],
                external_textures: &[],
            };

            let bg = unsafe { device.raw().create_bind_group(&bg_desc) }
                .map_err(DeviceError::from_hal)
                .map_err(RenderPassErrorInner::Device)?;

            unsafe {
                encoder.set_bind_group(inner.pipeline_layout.as_ref(), 0, bg.as_ref(), &[])
            };
            unsafe {
                encoder.set_immediates(
                    inner.pipeline_layout.as_ref(),
                    0,
                    &[
                        p.max_count,
                        p.stride_u32,
                        src_remainder_u32,
                        count_remainder_u32,
                    ],
                )
            };

            let wg_count = p.max_count.div_ceil(64);
            unsafe { encoder.dispatch_workgroups([wg_count, 1, 1]) };

            resources.bind_groups.push(bg);
        }

        unsafe { encoder.end_compute_pass() };

        {
            let mut barriers: Vec<hal::BufferBarrier<dyn hal::DynBuffer>> = Vec::new();

            for p in &pending {
                let src_buf = p.src_buffer.try_raw(snatch_guard).map_err(|_| {
                    RenderPassErrorInner::InvalidResource(InvalidResourceError(
                        p.src_buffer.error_ident(),
                    ))
                })?;
                let count_buf = p.count_buffer.try_raw(snatch_guard).map_err(|_| {
                    RenderPassErrorInner::InvalidResource(InvalidResourceError(
                        p.count_buffer.error_ident(),
                    ))
                })?;
                let temp_buffer = resources.get_temp_buffer(p.temp_buffer_index);
                barriers.push(hal::BufferBarrier {
                    buffer: src_buf,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::STORAGE_READ_ONLY,
                        to: wgt::BufferUses::INDIRECT,
                    },
                });
                barriers.push(hal::BufferBarrier {
                    buffer: count_buf,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::STORAGE_READ_ONLY,
                        to: wgt::BufferUses::INDIRECT,
                    },
                });
                barriers.push(hal::BufferBarrier {
                    buffer: temp_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::STORAGE_READ_WRITE,
                        to: wgt::BufferUses::INDIRECT,
                    },
                });
            }

            unsafe { encoder.transition_buffers(&barriers) };
        }

        Ok(())
    }
}

impl MultiDrawEmulationInner {
    fn new(
        device: &dyn hal::DynDevice,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<Self, CreateMultiDrawEmulationPipelineError> {
        let src = include_str!("multi_draw_count_emulation.wgsl");

        #[cfg(feature = "wgsl")]
        let module = naga::front::wgsl::parse_str(src).map_err(|inner| {
            CreateShaderModuleError::Parsing(naga::error::ShaderError {
                source: src.to_string(),
                label: None,
                inner: Box::new(inner),
            })
        })?;

        #[cfg(not(feature = "wgsl"))]
        #[allow(clippy::diverging_sub_expression)]
        let module = panic!("Multi-draw emulation requires the wgsl feature flag to be enabled!");

        let info = crate::device::create_validator(
            wgt::Features::IMMEDIATES,
            wgt::DownlevelFlags::empty(),
            naga::valid::ValidationFlags::all(),
        )
        .validate(&module)
        .map_err(|inner| {
            CreateShaderModuleError::Validation(naga::error::ShaderError {
                source: src.to_string(),
                label: None,
                inner: Box::new(inner),
            })
        })?;

        let hal_shader = hal::ShaderInput::Naga(hal::NagaShader {
            module: Cow::Owned(module),
            info,
            debug_source: None,
        });
        let hal_desc = hal::ShaderModuleDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw emulation shader module"),
                instance_flags,
            ),
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };
        let shader_module =
            unsafe { device.create_shader_module(&hal_desc, hal_shader) }.map_err(|error| {
                match error {
                    hal::ShaderError::Device(error) => {
                        CreateShaderModuleError::Device(DeviceError::from_hal(error))
                    }
                    hal::ShaderError::Compilation(ref msg) => {
                        log::error!("Shader error: {msg}");
                        CreateShaderModuleError::Generation
                    }
                }
            })?;

        let bgl_desc = hal::BindGroupLayoutDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw emulation bind group layout"),
                instance_flags,
            ),
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[
                wgt::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        };
        let bgl = unsafe {
            device
                .create_bind_group_layout(&bgl_desc)
                .map_err(DeviceError::from_hal)?
        };

        let pipeline_layout_desc = hal::PipelineLayoutDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw emulation pipeline layout"),
                instance_flags,
            ),
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[Some(bgl.as_ref())],
            immediate_size: 16,
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_desc)
                .map_err(DeviceError::from_hal)?
        };

        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: hal_label(
                Some("(wgpu internal) Multi-draw emulation pipeline"),
                instance_flags,
            ),
            layout: pipeline_layout.as_ref(),
            stage: hal::ProgrammableStage {
                module: shader_module.as_ref(),
                entry_point: "main",
                constants: &hashbrown::HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        };
        let pipeline =
            unsafe { device.create_compute_pipeline(&pipeline_desc) }.map_err(|err| match err {
                hal::PipelineError::Device(error) => {
                    CreateComputePipelineError::Device(DeviceError::from_hal(error))
                }
                hal::PipelineError::Linkage(_stages, msg) => {
                    CreateComputePipelineError::Internal(msg)
                }
                hal::PipelineError::EntryPoint(_stage) => CreateComputePipelineError::Internal(
                    crate::device::ENTRYPOINT_FAILURE_ERROR.to_string(),
                ),
                hal::PipelineError::PipelineConstants(_, error) => {
                    CreateComputePipelineError::PipelineConstants(error)
                }
            })?;

        Ok(Self {
            module: shader_module,
            bgl,
            pipeline_layout,
            pipeline,
            temp_pool: Mutex::new(rank::MULTI_DRAW_EMULATION_TEMP_POOL, Vec::new()),
        })
    }
}

impl MultiDrawResources {
    pub(crate) fn new(device: Arc<Device>) -> Self {
        MultiDrawResources {
            device,
            temp_entries: Vec::new(),
            bind_groups: Vec::new(),
        }
    }

    pub(crate) fn get_temp_buffer(&self, index: usize) -> &dyn hal::DynBuffer {
        self.temp_entries.get(index).unwrap().buffer.as_ref()
    }

    pub(crate) fn acquire_temp_entry(
        &mut self,
        size: u64,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<usize, DeviceError> {
        let emulation = self.device.multi_draw_emulation.as_ref().unwrap();
        let entry = emulation.acquire_temp_entry(self.device.raw(), size, instance_flags)?;
        let index = self.temp_entries.len();
        self.temp_entries.push(entry);
        Ok(index)
    }
}

impl Drop for MultiDrawResources {
    fn drop(&mut self) {
        let raw = self.device.raw();
        for bg in self.bind_groups.drain(..) {
            unsafe { raw.destroy_bind_group(bg) };
        }
        if let Some(emulation) = self.device.multi_draw_emulation.as_ref() {
            emulation.release_temp_entries(self.temp_entries.drain(..));
        }
    }
}
