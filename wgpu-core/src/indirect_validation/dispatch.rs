use super::CreateIndirectValidationPipelineError;
use crate::{
    device::DeviceError,
    hal_label,
    pipeline::{CreatePipelineError, CreateShaderModuleError},
};
use alloc::{boxed::Box, format, string::String};
use core::num::NonZeroU64;
use scopeguard::{guard, ScopeGuard};

/// This machinery requires the following limits:
///
/// - max_bind_groups: 2,
/// - max_dynamic_storage_buffers_per_pipeline_layout: 1,
/// - max_storage_buffers_per_shader_stage: 2,
/// - max_storage_buffer_binding_size: 3 * min_storage_buffer_offset_alignment,
/// - max_immediate_size: 4,
/// - max_compute_invocations_per_workgroup 1
///
/// These are all indirectly satisfied by `DownlevelFlags::INDIRECT_EXECUTION`, which is also
/// required for this module's functionality to work.
#[derive(Debug)]
pub(crate) struct Dispatch {
    src_bind_group_layout: Box<dyn hal::DynBindGroupLayout>,
    dispatch: ValidationPipeline,
    /// Validates the arguments of indirect trace rays calls. Only created if
    /// `Features::EXPERIMENTAL_RAY_TRACING_PIPELINES` is enabled.
    trace_rays: Option<ValidationPipeline>,
}

/// A pipeline that reads 3 `u32`s from the source buffer and writes validated arguments to its
/// own destination buffer.
#[derive(Debug)]
struct ValidationPipeline {
    module: Box<dyn hal::DynShaderModule>,
    dst_bind_group_layout: Box<dyn hal::DynBindGroupLayout>,
    pipeline_layout: Box<dyn hal::DynPipelineLayout>,
    pipeline: Box<dyn hal::DynComputePipeline>,
    dst_buffer: Box<dyn hal::DynBuffer>,
    dst_bind_group: Box<dyn hal::DynBindGroup>,
}

pub struct Params<'a> {
    pub pipeline_layout: &'a dyn hal::DynPipelineLayout,
    pub pipeline: &'a dyn hal::DynComputePipeline,
    pub dst_buffer: &'a dyn hal::DynBuffer,
    pub dst_bind_group: &'a dyn hal::DynBindGroup,
    pub aligned_offset: u64,
    pub offset_remainder: u64,
}

// SAFETY: The value we are passing to `new_unchecked` is not zero, so this is safe.
const SRC_BUFFER_SIZE: NonZeroU64 = NonZeroU64::new(size_of::<u32>() as u64 * 3).unwrap();

impl Dispatch {
    pub(super) fn new(
        device: &dyn hal::DynDevice,
        instance_flags: wgt::InstanceFlags,
        limits: &wgt::Limits,
        features: &wgt::Features,
    ) -> Result<Self, CreateIndirectValidationPipelineError> {
        let src_bind_group_layout_desc = hal::BindGroupLayoutDescriptor {
            label: hal_label(
                Some("(wgpu internal) Indirect dispatch validation source bind group layout"),
                instance_flags,
            ),
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(SRC_BUFFER_SIZE),
                },
                count: None,
            }],
        };
        let src_bind_group_layout = unsafe {
            device
                .create_bind_group_layout(&src_bind_group_layout_desc)
                .map_err(DeviceError::from_hal)?
        };
        let src_bind_group_layout = guard(src_bind_group_layout, |bgl| unsafe {
            device.destroy_bind_group_layout(bgl)
        });

        let max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;

        let dispatch_shader = format!(
            "
            @group(0) @binding(0)
            var<storage, read_write> dst: array<u32, 6>;
            @group(1) @binding(0)
            var<storage, read> src: array<u32>;
            struct OffsetPc {{
                inner: u32,
            }}
            var<immediate> offset: OffsetPc;

            @compute @workgroup_size(1)
            fn main() {{
                let src = vec3(src[offset.inner], src[offset.inner + 1], src[offset.inner + 2]);
                let max_compute_workgroups_per_dimension = {max_compute_workgroups_per_dimension}u;
                if (
                    src.x > max_compute_workgroups_per_dimension ||
                    src.y > max_compute_workgroups_per_dimension ||
                    src.z > max_compute_workgroups_per_dimension
                ) {{
                    dst = array(0u, 0u, 0u, 0u, 0u, 0u);
                }} else {{
                    dst = array(src.x, src.y, src.z, src.x, src.y, src.z);
                }}
            }}
        "
        );

        // SAFETY: The value we are passing to `new_unchecked` is not zero, so this is safe.
        const DST_BUFFER_SIZE: NonZeroU64 = NonZeroU64::new(SRC_BUFFER_SIZE.get() * 2).unwrap();

        let dispatch = ValidationPipeline::new(
            device,
            instance_flags,
            src_bind_group_layout.as_ref(),
            "dispatch",
            dispatch_shader,
            DST_BUFFER_SIZE,
        )?;
        let dispatch = guard(dispatch, |dispatch| dispatch.dispose(device));

        let trace_rays = if features.contains(wgt::Features::EXPERIMENTAL_RAY_TRACING_PIPELINES) {
            let [max_x, max_y, max_z] = crate::command::max_trace_rays_dimensions(limits);
            let max_total = limits.max_ray_dispatch_count;

            let trace_rays_shader = format!(
                "
                @group(0) @binding(0)
                var<storage, read_write> dst: array<u32, 3>;
                @group(1) @binding(0)
                var<storage, read> src: array<u32>;
                struct OffsetPc {{
                    inner: u32,
                }}
                var<immediate> offset: OffsetPc;

                @compute @workgroup_size(1)
                fn main() {{
                    let src = vec3(src[offset.inner], src[offset.inner + 1], src[offset.inner + 2]);
                    let max_total = {max_total}u;
                    var valid = all(src <= vec3({max_x}u, {max_y}u, {max_z}u));
                    // Check `src.x * src.y * src.z <= max_total` without overflowing.
                    if (valid && src.x != 0u) {{
                        valid = src.y <= max_total / src.x;
                    }}
                    if (valid && src.x * src.y != 0u) {{
                        valid = src.z <= max_total / (src.x * src.y);
                    }}
                    if (valid) {{
                        dst = array(src.x, src.y, src.z);
                    }} else {{
                        dst = array(0u, 0u, 0u);
                    }}
                }}
            "
            );

            Some(ValidationPipeline::new(
                device,
                instance_flags,
                src_bind_group_layout.as_ref(),
                "trace rays",
                trace_rays_shader,
                SRC_BUFFER_SIZE,
            )?)
        } else {
            None
        };

        // Error returns after we start consuming guards could bypass resource cleanup.
        #[deny(clippy::question_mark_used)]
        Ok(Self {
            src_bind_group_layout: ScopeGuard::into_inner(src_bind_group_layout),
            dispatch: ScopeGuard::into_inner(dispatch),
            trace_rays,
        })
    }

    /// `Ok(None)` will only be returned if `buffer_size` is `0`.
    pub(super) fn create_src_bind_group(
        &self,
        device: &dyn hal::DynDevice,
        limits: &wgt::Limits,
        buffer_size: u64,
        buffer: &dyn hal::DynBuffer,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<Option<Box<dyn hal::DynBindGroup>>, DeviceError> {
        let binding_size = calculate_src_buffer_binding_size(buffer_size, limits);
        let Some(binding_size) = NonZeroU64::new(binding_size) else {
            return Ok(None);
        };
        let hal_desc = hal::BindGroupDescriptor {
            label: hal_label(
                Some("(wgpu internal) Indirect dispatch validation source bind group"),
                instance_flags,
            ),
            layout: self.src_bind_group_layout.as_ref(),
            entries: &[hal::BindGroupEntry {
                binding: 0,
                resource_index: 0,
                count: 1,
            }],
            // SAFETY: We calculated the binding size to fit within the buffer.
            buffers: &[hal::BufferBinding::new_unchecked(buffer, 0, binding_size)],
            samplers: &[],
            textures: &[],
            acceleration_structures: &[],
            external_textures: &[],
        };
        unsafe {
            device
                .create_bind_group(&hal_desc)
                .map(Some)
                .map_err(DeviceError::from_hal)
        }
    }

    /// Parameters for validating the arguments of an indirect compute dispatch.
    ///
    /// The destination buffer holds the validated arguments at offset 0.
    pub fn params<'a>(&'a self, limits: &wgt::Limits, offset: u64, buffer_size: u64) -> Params<'a> {
        self.dispatch.params(limits, offset, buffer_size)
    }

    /// Parameters for validating the arguments of an indirect trace rays call.
    ///
    /// The destination buffer holds the validated arguments at offset 0. Returns `None` if
    /// `Features::EXPERIMENTAL_RAY_TRACING_PIPELINES` is not enabled.
    pub fn trace_rays_params<'a>(
        &'a self,
        limits: &wgt::Limits,
        offset: u64,
        buffer_size: u64,
    ) -> Option<Params<'a>> {
        self.trace_rays
            .as_ref()
            .map(|trace_rays| trace_rays.params(limits, offset, buffer_size))
    }

    pub(super) fn dispose(self, device: &dyn hal::DynDevice) {
        let Dispatch {
            src_bind_group_layout,
            dispatch,
            trace_rays,
        } = self;

        if let Some(trace_rays) = trace_rays {
            trace_rays.dispose(device);
        }
        dispatch.dispose(device);
        unsafe {
            device.destroy_bind_group_layout(src_bind_group_layout);
        }
    }
}

impl ValidationPipeline {
    fn new(
        device: &dyn hal::DynDevice,
        instance_flags: wgt::InstanceFlags,
        src_bind_group_layout: &dyn hal::DynBindGroupLayout,
        name: &str,
        src: String,
        dst_buffer_size: NonZeroU64,
    ) -> Result<Self, CreateIndirectValidationPipelineError> {
        #[cfg(feature = "wgsl")]
        let module = naga::front::wgsl::parse_str(&src).map_err(|inner| {
            CreateShaderModuleError::Parsing(naga::error::ShaderError {
                source: src.clone(),
                label: None,
                inner: Box::new(inner),
            })
        })?;
        #[cfg(not(feature = "wgsl"))]
        #[allow(clippy::diverging_sub_expression)]
        let module = panic!("Indirect validation requires the wgsl feature flag to be enabled!");

        let info = crate::device::create_validator(
            wgt::Features::IMMEDIATES,
            wgt::DownlevelFlags::empty(),
            naga::valid::ValidationFlags::all(),
        )
        .validate(&module)
        .map_err(|inner| {
            CreateShaderModuleError::Validation(naga::error::ShaderError {
                source: src,
                label: None,
                inner,
            })
        })?;
        let hal_shader = hal::ShaderInput::Naga(hal::NagaShader {
            module: alloc::borrow::Cow::Owned(module),
            info,
            debug_source: None,
        });
        let module_label = format!("(wgpu internal) Indirect {name} validation shader module");
        let hal_desc = hal::ShaderModuleDescriptor {
            label: hal_label(Some(module_label.as_str()), instance_flags),
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };
        let module =
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
        let module = guard(module, |module| unsafe {
            device.destroy_shader_module(module)
        });

        let dst_bind_group_layout_label =
            format!("(wgpu internal) Indirect {name} validation destination bind group layout");
        let dst_bind_group_layout_desc = hal::BindGroupLayoutDescriptor {
            label: hal_label(Some(dst_bind_group_layout_label.as_str()), instance_flags),
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(dst_buffer_size),
                },
                count: None,
            }],
        };
        let dst_bind_group_layout = unsafe {
            device
                .create_bind_group_layout(&dst_bind_group_layout_desc)
                .map_err(DeviceError::from_hal)?
        };
        let dst_bind_group_layout = guard(dst_bind_group_layout, |bgl| unsafe {
            device.destroy_bind_group_layout(bgl)
        });

        let pipeline_layout_label =
            format!("(wgpu internal) Indirect {name} validation pipeline layout");
        let pipeline_layout_desc = hal::PipelineLayoutDescriptor {
            label: hal_label(Some(pipeline_layout_label.as_str()), instance_flags),
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[
                Some(dst_bind_group_layout.as_ref()),
                Some(src_bind_group_layout),
            ],
            immediate_size: 4,
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_desc)
                .map_err(DeviceError::from_hal)?
        };
        let pipeline_layout = guard(pipeline_layout, |pipeline_layout| unsafe {
            device.destroy_pipeline_layout(pipeline_layout)
        });

        let pipeline_label = format!("(wgpu internal) Indirect {name} validation pipeline");
        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: hal_label(Some(pipeline_label.as_str()), instance_flags),
            layout: pipeline_layout.as_ref(),
            stage: hal::ProgrammableStage {
                module: module.as_ref(),
                entry_point: "main",
                constants: &Default::default(),
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        };
        let pipeline = unsafe { device.create_compute_pipeline(&pipeline_desc) }
            .map_err(CreatePipelineError::from_hal_without_device_loss)?;
        let pipeline = guard(pipeline, |pipeline| unsafe {
            device.destroy_compute_pipeline(pipeline)
        });

        let dst_buffer_label =
            format!("(wgpu internal) Indirect {name} validation destination buffer");
        let dst_buffer_desc = hal::BufferDescriptor {
            label: hal_label(Some(dst_buffer_label.as_str()), instance_flags),
            size: dst_buffer_size.get(),
            usage: wgt::BufferUses::INDIRECT | wgt::BufferUses::STORAGE_READ_WRITE,
            memory_flags: hal::MemoryFlags::empty(),
        };
        let (dst_buffer, _) =
            unsafe { device.create_buffer(&dst_buffer_desc) }.map_err(DeviceError::from_hal)?;
        let dst_buffer = guard(dst_buffer, |buffer| unsafe {
            device.destroy_buffer(buffer)
        });

        let dst_bind_group_label =
            format!("(wgpu internal) Indirect {name} validation destination bind group");
        let dst_bind_group_desc = hal::BindGroupDescriptor {
            label: hal_label(Some(dst_bind_group_label.as_str()), instance_flags),
            layout: dst_bind_group_layout.as_ref(),
            entries: &[hal::BindGroupEntry {
                binding: 0,
                resource_index: 0,
                count: 1,
            }],
            // SAFETY: We just created the buffer with this size.
            buffers: &[hal::BufferBinding::new_unchecked(
                dst_buffer.as_ref(),
                0,
                dst_buffer_size,
            )],
            samplers: &[],
            textures: &[],
            acceleration_structures: &[],
            external_textures: &[],
        };
        let dst_bind_group = unsafe {
            device
                .create_bind_group(&dst_bind_group_desc)
                .map_err(DeviceError::from_hal)
        }?;

        // Error returns after we start consuming guards could bypass resource cleanup.
        #[deny(clippy::question_mark_used)]
        Ok(Self {
            module: ScopeGuard::into_inner(module),
            dst_bind_group_layout: ScopeGuard::into_inner(dst_bind_group_layout),
            pipeline_layout: ScopeGuard::into_inner(pipeline_layout),
            pipeline: ScopeGuard::into_inner(pipeline),
            dst_buffer: ScopeGuard::into_inner(dst_buffer),
            dst_bind_group,
        })
    }

    fn params<'a>(&'a self, limits: &wgt::Limits, offset: u64, buffer_size: u64) -> Params<'a> {
        // The offset we receive is only required to be aligned to 4 bytes.
        //
        // Binding offsets and dynamic offsets are required to be aligned to
        // min_storage_buffer_offset_alignment (256 bytes by default).
        //
        // So, we work around this limitation by calculating an aligned offset
        // and pass the remainder through a immediate data.
        //
        // We could bind the whole buffer and only have to pass the offset
        // through a immediate data but we might run into the
        // max_storage_buffer_binding_size limit.
        //
        // See the inner docs of `calculate_src_buffer_binding_size` to
        // see how we get the appropriate `binding_size`.
        let alignment = limits.min_storage_buffer_offset_alignment as u64;
        let binding_size = calculate_src_buffer_binding_size(buffer_size, limits);
        let aligned_offset = offset - offset % alignment;
        // This works because `binding_size` is either `buffer_size` or `alignment * 2 + buffer_size % alignment`.
        let max_aligned_offset = buffer_size - binding_size;
        let aligned_offset = aligned_offset.min(max_aligned_offset);
        let offset_remainder = offset - aligned_offset;

        Params {
            pipeline_layout: self.pipeline_layout.as_ref(),
            pipeline: self.pipeline.as_ref(),
            dst_buffer: self.dst_buffer.as_ref(),
            dst_bind_group: self.dst_bind_group.as_ref(),
            aligned_offset,
            offset_remainder,
        }
    }

    fn dispose(self, device: &dyn hal::DynDevice) {
        let ValidationPipeline {
            module,
            dst_bind_group_layout,
            pipeline_layout,
            pipeline,
            dst_buffer,
            dst_bind_group,
        } = self;

        unsafe {
            device.destroy_bind_group(dst_bind_group);
            device.destroy_buffer(dst_buffer);
            device.destroy_compute_pipeline(pipeline);
            device.destroy_pipeline_layout(pipeline_layout);
            device.destroy_bind_group_layout(dst_bind_group_layout);
            device.destroy_shader_module(module);
        }
    }
}

fn calculate_src_buffer_binding_size(buffer_size: u64, limits: &wgt::Limits) -> u64 {
    let alignment = limits.min_storage_buffer_offset_alignment as u64;

    // We need to choose a binding size that can address all possible sets of 12 contiguous bytes in the buffer taking
    // into account that the dynamic offset needs to be a multiple of `min_storage_buffer_offset_alignment`.

    // Given the know variables: `offset`, `buffer_size`, `alignment` and the rule `offset + 12 <= buffer_size`.

    // Let `chunks = floor(buffer_size / alignment)`.
    // Let `chunk` be the interval `[0, chunks]`.
    // Let `offset = alignment * chunk + r` where `r` is the interval [0, alignment - 4].
    // Let `binding` be the interval `[offset, offset + 12]`.
    // Let `aligned_offset = alignment * chunk`.
    // Let `aligned_binding` be the interval `[aligned_offset, aligned_offset + r + 12]`.
    // Let `aligned_binding_size = r + 12 = [12, alignment + 8]`.
    // Let `min_aligned_binding_size = alignment + 8`.

    // `min_aligned_binding_size` is the minimum binding size required to address all 12 contiguous bytes in the buffer
    // but the last aligned_offset + min_aligned_binding_size might overflow the buffer. In order to avoid this we must
    // pick a larger `binding_size` that satisfies: `last_aligned_offset + binding_size = buffer_size` and
    // `binding_size >= min_aligned_binding_size`.

    // Let `buffer_size = alignment * chunks + sr` where `sr` is the interval [0, alignment - 4].
    // Let `last_aligned_offset = alignment * (chunks - u)` where `u` is the interval [0, chunks].
    // => `binding_size = buffer_size - last_aligned_offset`
    // => `binding_size = alignment * chunks + sr - alignment * (chunks - u)`
    // => `binding_size = alignment * chunks + sr - alignment * chunks + alignment * u`
    // => `binding_size = sr + alignment * u`
    // => `min_aligned_binding_size <= sr + alignment * u`
    // => `alignment + 8 <= sr + alignment * u`
    // => `u` must be at least 2
    // => `binding_size = sr + alignment * 2`

    let binding_size = 2 * alignment + (buffer_size % alignment);
    binding_size.min(buffer_size)
}
