use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use core::{mem::ManuallyDrop, num::NonZeroU64};
use hal::BufferDescriptor;

use arrayvec::ArrayVec;
use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::pipeline::{
    ImplicitLayoutError, LateSizedBufferGroup, PipelineCache, ProgrammableStageDescriptor,
    ShaderModule,
};
use crate::{
    binding_model::{BindGroupLayout, GetBindGroupLayoutError, PipelineLayout},
    device::{Device, DeviceError, MissingFeatures},
    resource::{InvalidResourceError, Labeled, ResourceState, TrackingData},
    resource_log,
    validation::{self},
    Label,
};

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateRayTracingPipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("Error matching {stage:?} shader requirements against the pipeline")]
    Stage {
        stage: wgt::ShaderStages,
        #[source]
        error: validation::StageError,
    },
    #[error("In the provided shader, the type given for group {group} binding {binding} has a size of {size}. As the device does not support `DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`, the type must have a size that is a multiple of 16 bytes.")]
    UnalignedShader { group: u32, binding: u32, size: u64 },
    #[error("Internal error in {stage:?} shader: {error}")]
    Internal {
        stage: wgt::ShaderStages,
        error: String,
    },
    #[error("Pipeline constant error in {stage:?} shader: {error}")]
    PipelineConstants {
        stage: wgt::ShaderStages,
        error: String,
    },
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error("The number of intersection shaders {0} is greater than 2^24")]
    TooManyIntersectionGroups(usize),
    #[error(
        "The ray recursion depth {0} is greater than the limit Limits::max_ray_recursion_depth {1}"
    )]
    TooHighRayRecursionDepth(u32, u32),
}

impl WebGpuError for CreateRayTracingPipelineError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),

            Self::Internal { .. } => ErrorType::Internal,

            Self::Implicit(_)
            | Self::Stage { .. }
            | Self::UnalignedShader { .. }
            | Self::PipelineConstants { .. }
            | Self::TooManyIntersectionGroups(..)
            | Self::TooHighRayRecursionDepth(..) => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RayTracingIntersectionDescriptor<'a, SM = Arc<ShaderModule>> {
    Triangle {
        closest_hit: ProgrammableStageDescriptor<'a, SM>,
        any_hit: Option<ProgrammableStageDescriptor<'a, SM>>,
    },
}

/// Describes a ray tracing pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RayTracingPipelineDescriptor<
    'a,
    PLL = Arc<PipelineLayout>,
    SM = Arc<ShaderModule>,
    PLC = Arc<PipelineCache>,
> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<PLL>,
    /// The ray generation stage descriptor
    pub ray_generation: ProgrammableStageDescriptor<'a, SM>,
    /// The miss stage descriptor
    pub miss: ProgrammableStageDescriptor<'a, SM>,
    /// All the descriptors for intersection functions
    pub intersections: Vec<RayTracingIntersectionDescriptor<'a, SM>>,
    /// The maximum amount entry points are allowed to recurse
    pub max_recursion_depth: u32,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<PLC>,
}

/// The type of intersection of a index into `RayTracingPipelineDescriptor::intersections`
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RayTracingIntersectionType {
    /// The intersection is to be used for triangles.
    Triangle,
    /// The intersection is to be used for AABBs.
    AABB,
}

/// Metal's shader binding data is opaque, but Vulkan's and DX12's has opaque data
/// but a non-opaque storage mechanism, so each require separate codepaths.
/// Therefore, this is a semi-opaque structure because if metal gets ray tracing pipelines,
/// this will need to turn into an enum so it shouldn't have other code
/// tangled with it.
#[derive(Debug)]
pub struct ShaderBindingData {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    pub(crate) device: Arc<Device>,
    pub(crate) num_intersection_groups: u32,
    pub(crate) miss_offset: wgt::BufferAddress,
    pub(crate) intersection_offset: wgt::BufferAddress,
}

impl ShaderBindingData {
    /// Panics if `num_intersection_groups` is above u32::MAX
    pub(crate) fn from_raw_pipeline(
        device: Arc<Device>,
        pipeline: &dyn hal::DynRayTracingPipeline,
        num_intersection_groups: usize,
    ) -> Result<Arc<Self>, CreateRayTracingPipelineError> {
        let mut base_data = Vec::new();

        let ray_generation_data = unsafe {
            device
                .raw()
                .get_raytracing_pipeline_group_data(pipeline, 0..1)
        }
        .map_err(|e| CreateRayTracingPipelineError::Device(device.handle_hal_error(e)))?;

        base_data.extend_from_slice(&ray_generation_data);

        let padded_miss_offset = (base_data.len() as wgt::BufferAddress).next_multiple_of(
            device.alignments.ray_tracing_pipeline_data_offset_alignment as wgt::BufferAddress,
        );

        base_data.resize(padded_miss_offset as _, 0);

        let miss_data = unsafe {
            device
                .raw()
                .get_raytracing_pipeline_group_data(pipeline, 1..2)
        }
        .map_err(|e| CreateRayTracingPipelineError::Device(device.handle_hal_error(e)))?;

        base_data.extend_from_slice(&miss_data);

        let padded_intersection_offset = (base_data.len() as wgt::BufferAddress).next_multiple_of(
            device.alignments.ray_tracing_pipeline_data_offset_alignment as wgt::BufferAddress,
        );

        base_data.resize(padded_intersection_offset as _, 0);

        if num_intersection_groups != 0 {
            let intersection_data = unsafe {
                device.raw().get_raytracing_pipeline_group_data(
                    pipeline,
                    2..(2 + num_intersection_groups as u32),
                )
            }
            .map_err(|e| CreateRayTracingPipelineError::Device(device.handle_hal_error(e)))?;

            base_data.extend_from_slice(&intersection_data);
        }

        // Don't need the exact allocated size as it will be grater than this.
        let (buffer, _) = unsafe {
            device.raw().create_buffer(&BufferDescriptor {
                label: None,
                size: base_data.len() as _,
                usage: wgt::BufferUses::RAY_TRACING_PIPELINE_SHADER_DATA
                    | wgt::BufferUses::COPY_DST,
                memory_flags: hal::MemoryFlags::PREFER_COHERENT,
            })
        }
        .map_err(|e| CreateRayTracingPipelineError::Device(device.handle_hal_error(e)))?;

        let sbd = Arc::new(Self {
            raw: ManuallyDrop::new(buffer),
            device: device.clone(),
            num_intersection_groups: u32::try_from(num_intersection_groups).unwrap(),
            miss_offset: padded_miss_offset,
            intersection_offset: padded_intersection_offset,
        });

        // If there is no queue anymore, the ray tracing pipeline can't be accessed, so we don't have to worry about UB from uninitialized values
        if let Some(queue) = device.get_queue() {
            let mut staging = crate::resource::StagingBuffer::new(
                &device,
                NonZeroU64::new(base_data.len() as _).expect(
                    "The total number of groups is always greater than zero, \
                         and `ray_tracing_pipeline_group_data_size` must be too.",
                ),
            )?;

            staging.write_exact(&base_data);

            let staging_buf = staging.flush();

            let mut writes = queue.pending_writes.lock();
            let encoder = writes.activate();
            unsafe {
                encoder.transition_buffers(&[
                    hal::BufferBarrier {
                        buffer: staging_buf.raw(),
                        usage: hal::StateTransition {
                            from: wgt::BufferUses::MAP_WRITE,
                            to: wgt::BufferUses::COPY_SRC,
                        },
                    },
                    hal::BufferBarrier {
                        buffer: sbd.raw.as_ref(),
                        usage: hal::StateTransition {
                            from: wgt::BufferUses::empty(),
                            to: wgt::BufferUses::COPY_DST,
                        },
                    },
                ]);
                encoder.copy_buffer_to_buffer(
                    staging_buf.raw(),
                    sbd.raw.as_ref(),
                    &[hal::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: NonZeroU64::new(base_data.len() as _)
                            .expect("Already checked size isn't zero."),
                    }],
                );
                encoder.transition_buffers(&[hal::BufferBarrier {
                    buffer: sbd.raw.as_ref(),
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::COPY_DST,
                        to: wgt::BufferUses::RAY_TRACING_PIPELINE_SHADER_DATA,
                    },
                }]);
            };

            writes.consume(staging_buf);
            writes.use_shader_binding_data(&sbd);
        }

        Ok(sbd)
    }
}

impl Drop for ShaderBindingData {
    fn drop(&mut self) {
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_buffer(raw);
        }
    }
}

#[derive(Debug)]
pub(crate) struct RayTracingPipelineState {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynRayTracingPipeline>>,
    pub(crate) layout: Arc<PipelineLayout>,
    pub(crate) shader_binding_data: Arc<ShaderBindingData>,
    pub(crate) _shader_modules: Vec<Arc<ShaderModule>>,
    pub(crate) intersection_types: Arc<[RayTracingIntersectionType]>,
}

#[derive(Debug)]
pub struct RayTracingPipeline {
    pub(crate) state: ResourceState<RayTracingPipelineState>,
    pub(crate) device: Arc<Device>,
    pub(crate) late_sized_buffer_groups: ArrayVec<LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }>,
    pub(crate) immediate_slots_required: naga::valid::ImmediateSlots,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());

        #[cfg(feature = "trace")]
        {
            use crate::device::trace;
            if let Some(t) = self.device.trace.lock().as_mut() {
                t.add(trace::Action::DropRayTracingPipeline(unsafe {
                    trace::to_trace(self)
                }));
            }
        }

        let ResourceState::Valid(state) = &mut self.state else {
            return;
        };

        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut state.raw) };
        unsafe {
            self.device.raw().destroy_ray_tracing_pipeline(raw);
        }
    }
}

crate::impl_resource_type!(RayTracingPipeline);
crate::impl_labeled!(RayTracingPipeline);
crate::impl_parent_device!(RayTracingPipeline);
crate::impl_storage_item!(RayTracingPipeline);
crate::impl_trackable!(RayTracingPipeline);

impl RayTracingPipeline {
    pub(crate) fn raw(&self) -> Result<&dyn hal::DynRayTracingPipeline, InvalidResourceError> {
        let ResourceState::Valid(state) = &self.state else {
            return Err(InvalidResourceError(self.error_ident()));
        };
        Ok(state.raw.as_ref())
    }

    pub(crate) fn invalid(device: Arc<Device>, label: String) -> Arc<Self> {
        Arc::new(Self {
            tracking_data: TrackingData::new(device.tracker_indices.ray_tracing_pipelines.clone()),
            state: ResourceState::Invalid,
            device,
            immediate_slots_required: naga::valid::ImmediateSlots::default(),
            late_sized_buffer_groups: ArrayVec::new(),
            label,
        })
    }

    pub(crate) fn check_valid(&self) -> Result<(), InvalidResourceError> {
        let ResourceState::Valid(_) = &self.state else {
            return Err(InvalidResourceError(self.error_ident()));
        };
        Ok(())
    }

    pub(crate) fn layout(&self) -> Result<&Arc<PipelineLayout>, InvalidResourceError> {
        let ResourceState::Valid(state) = &self.state else {
            return Err(InvalidResourceError(self.error_ident()));
        };
        Ok(&state.layout)
    }

    pub(crate) fn shader_binding_data(&self) -> Result<&ShaderBindingData, InvalidResourceError> {
        let ResourceState::Valid(state) = &self.state else {
            return Err(InvalidResourceError(self.error_ident()));
        };
        Ok(&state.shader_binding_data)
    }

    pub(crate) fn intersection_types(
        &self,
    ) -> Result<&Arc<[RayTracingIntersectionType]>, InvalidResourceError> {
        let ResourceState::Valid(state) = &self.state else {
            return Err(InvalidResourceError(self.error_ident()));
        };
        Ok(&state.intersection_types)
    }

    pub fn get_bind_group_layout_inner(
        self: &Arc<Self>,
        index: u32,
    ) -> Result<Arc<BindGroupLayout>, GetBindGroupLayoutError> {
        self.layout()?.get_bind_group_layout(index, self.into())
    }

    pub fn get_bind_group_layout(
        self: &Arc<Self>,
        index: u32,
    ) -> (Arc<BindGroupLayout>, Option<GetBindGroupLayoutError>) {
        let (bgl, error) = match self.get_bind_group_layout_inner(index) {
            Ok(bgl) => (bgl, None),
            Err(e) => (
                BindGroupLayout::invalid(&self.device, String::new()),
                Some(e),
            ),
        };
        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.device.trace.lock() {
            use crate::device::trace;
            use trace::IntoTrace;
            trace.add(trace::Action::GetRayTracingPipelineBindGroupLayout {
                id: bgl.to_trace(),
                pipeline: self.to_trace(),
                index,
            });
        };
        (bgl, error)
    }
}
