use crate::{
    device::{DeviceError, MissingDownlevelFlags, MissingFeatures, SHADER_STAGE_COUNT},
    error::{ErrorFormatter, PrettyError},
    hub::{HalApi, Resource},
    id::{BindGroupLayoutId, BufferId, DeviceId, SamplerId, TextureId, TextureViewId, Valid},
    init_tracker::{BufferInitTrackerAction, TextureInitTrackerAction},
    track::{BindGroupStates, UsageConflict},
    validation::{MissingBufferUsageError, MissingTextureUsageError},
    FastHashMap, Label, LifeGuard, MultiRefCount, Stored,
};

use arrayvec::ArrayVec;

#[cfg(feature = "replay")]
use serde::Deserialize;
#[cfg(feature = "trace")]
use serde::Serialize;

use std::{borrow::Cow, ops::Range};

use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum BindGroupLayoutEntryError {
    #[error("cube dimension is not expected for texture storage")]
    StorageTextureCube,
    #[error("Read-write and read-only storage textures are not allowed by webgpu, they require the native only feature TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES")]
    StorageTextureReadWrite,
    #[error("arrays of bindings unsupported for this type of binding")]
    ArrayUnsupported,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

#[derive(Clone, Debug, Error)]
pub enum CreateBindGroupLayoutError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("conflicting binding at index {0}")]
    ConflictBinding(u32),
    #[error("binding {binding} entry is invalid")]
    Entry {
        binding: u32,
        #[source]
        error: BindGroupLayoutEntryError,
    },
    #[error(transparent)]
    TooManyBindings(BindingTypeMaxCountError),
}

//TODO: refactor this to move out `enum BindingError`.

#[derive(Clone, Debug, Error)]
pub enum CreateBindGroupError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("bind group layout is invalid")]
    InvalidLayout,
    #[error("buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("texture view {0:?} is invalid")]
    InvalidTextureView(TextureViewId),
    #[error("texture {0:?} is invalid")]
    InvalidTexture(TextureId),
    #[error("sampler {0:?} is invalid")]
    InvalidSampler(SamplerId),
    #[error(
        "binding count declared with at most {expected} items, but {actual} items were provided"
    )]
    BindingArrayPartialLengthMismatch { actual: usize, expected: usize },
    #[error(
        "binding count declared with exactly {expected} items, but {actual} items were provided"
    )]
    BindingArrayLengthMismatch { actual: usize, expected: usize },
    #[error("array binding provided zero elements")]
    BindingArrayZeroLength,
    #[error("bound buffer range {range:?} does not fit in buffer of size {size}")]
    BindingRangeTooLarge {
        buffer: BufferId,
        range: Range<wgt::BufferAddress>,
        size: u64,
    },
    #[error("buffer binding size {actual} is less than minimum {min}")]
    BindingSizeTooSmall {
        buffer: BufferId,
        actual: u64,
        min: u64,
    },
    #[error("buffer binding size is zero")]
    BindingZeroSize(BufferId),
    #[error("number of bindings in bind group descriptor ({actual}) does not match the number of bindings defined in the bind group layout ({expected})")]
    BindingsNumMismatch { actual: usize, expected: usize },
    #[error("binding {0} is used at least twice in the descriptor")]
    DuplicateBinding(u32),
    #[error("unable to find a corresponding declaration for the given binding {0}")]
    MissingBindingDeclaration(u32),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error("binding declared as a single item, but bind group is using it as an array")]
    SingleBindingExpected,
    #[error("buffer offset {0} does not respect device's requested `{1}` limit {2}")]
    UnalignedBufferOffset(wgt::BufferAddress, &'static str, u32),
    #[error(
        "buffer binding {binding} range {given} exceeds `max_*_buffer_binding_size` limit {limit}"
    )]
    BufferRangeTooLarge {
        binding: u32,
        given: u32,
        limit: u32,
    },
    #[error("binding {binding} has a different type ({actual:?}) than the one in the layout ({expected:?})")]
    WrongBindingType {
        // Index of the binding
        binding: u32,
        // The type given to the function
        actual: wgt::BindingType,
        // Human-readable description of expected types
        expected: &'static str,
    },
    #[error("texture binding {binding} expects multisampled = {layout_multisampled}, but given a view with samples = {view_samples}")]
    InvalidTextureMultisample {
        binding: u32,
        layout_multisampled: bool,
        view_samples: u32,
    },
    #[error("texture binding {binding} expects sample type = {layout_sample_type:?}, but given a view with format = {view_format:?}")]
    InvalidTextureSampleType {
        binding: u32,
        layout_sample_type: wgt::TextureSampleType,
        view_format: wgt::TextureFormat,
    },
    #[error("texture binding {binding} expects dimension = {layout_dimension:?}, but given a view with dimension = {view_dimension:?}")]
    InvalidTextureDimension {
        binding: u32,
        layout_dimension: wgt::TextureViewDimension,
        view_dimension: wgt::TextureViewDimension,
    },
    #[error("storage texture binding {binding} expects format = {layout_format:?}, but given a view with format = {view_format:?}")]
    InvalidStorageTextureFormat {
        binding: u32,
        layout_format: wgt::TextureFormat,
        view_format: wgt::TextureFormat,
    },
    #[error("storage texture bindings must have a single mip level, but given a view with mip_level_count = {mip_level_count:?} at binding {binding}")]
    InvalidStorageTextureMipLevelCount { binding: u32, mip_level_count: u32 },
    #[error("sampler binding {binding} expects comparison = {layout_cmp}, but given a sampler with comparison = {sampler_cmp}")]
    WrongSamplerComparison {
        binding: u32,
        layout_cmp: bool,
        sampler_cmp: bool,
    },
    #[error("sampler binding {binding} expects filtering = {layout_flt}, but given a sampler with filtering = {sampler_flt}")]
    WrongSamplerFiltering {
        binding: u32,
        layout_flt: bool,
        sampler_flt: bool,
    },
    #[error("bound texture views can not have both depth and stencil aspects enabled")]
    DepthStencilAspect,
    #[error("the adapter does not support read access for storages texture of format {0:?}")]
    StorageReadNotSupported(wgt::TextureFormat),
    #[error(transparent)]
    ResourceUsageConflict(#[from] UsageConflict),
}

impl PrettyError for CreateBindGroupError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::BindingZeroSize(id) => {
                fmt.buffer_label(&id);
            }
            Self::BindingRangeTooLarge { buffer, .. } => {
                fmt.buffer_label(&buffer);
            }
            Self::BindingSizeTooSmall { buffer, .. } => {
                fmt.buffer_label(&buffer);
            }
            Self::InvalidBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::InvalidTextureView(id) => {
                fmt.texture_view_label(&id);
            }
            Self::InvalidSampler(id) => {
                fmt.sampler_label(&id);
            }
            _ => {}
        };
    }
}

#[derive(Clone, Debug, Error)]
pub enum BindingZone {
    #[error("stage {0:?}")]
    Stage(wgt::ShaderStages),
    #[error("whole pipeline")]
    Pipeline,
}

#[derive(Clone, Debug, Error)]
#[error("too many bindings of type {kind:?} in {zone}, limit is {limit}, count was {count}")]
pub struct BindingTypeMaxCountError {
    pub kind: BindingTypeMaxCountErrorKind,
    pub zone: BindingZone,
    pub limit: u32,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub enum BindingTypeMaxCountErrorKind {
    DynamicUniformBuffers,
    DynamicStorageBuffers,
    SampledTextures,
    Samplers,
    StorageBuffers,
    StorageTextures,
    UniformBuffers,
}

#[derive(Debug, Default)]
pub(crate) struct PerStageBindingTypeCounter {
    vertex: u32,
    fragment: u32,
    compute: u32,
}

impl PerStageBindingTypeCounter {
    pub(crate) fn add(&mut self, stage: wgt::ShaderStages, count: u32) {
        if stage.contains(wgt::ShaderStages::VERTEX) {
            self.vertex += count;
        }
        if stage.contains(wgt::ShaderStages::FRAGMENT) {
            self.fragment += count;
        }
        if stage.contains(wgt::ShaderStages::COMPUTE) {
            self.compute += count;
        }
    }

    pub(crate) fn max(&self) -> (BindingZone, u32) {
        let max_value = self.vertex.max(self.fragment.max(self.compute));
        let mut stage = wgt::ShaderStages::NONE;
        if max_value == self.vertex {
            stage |= wgt::ShaderStages::VERTEX
        }
        if max_value == self.fragment {
            stage |= wgt::ShaderStages::FRAGMENT
        }
        if max_value == self.compute {
            stage |= wgt::ShaderStages::COMPUTE
        }
        (BindingZone::Stage(stage), max_value)
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.vertex = self.vertex.max(other.vertex);
        self.fragment = self.fragment.max(other.fragment);
        self.compute = self.compute.max(other.compute);
    }

    pub(crate) fn validate(
        &self,
        limit: u32,
        kind: BindingTypeMaxCountErrorKind,
    ) -> Result<(), BindingTypeMaxCountError> {
        let (zone, count) = self.max();
        if limit < count {
            Err(BindingTypeMaxCountError {
                kind,
                zone,
                limit,
                count,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BindingTypeMaxCountValidator {
    dynamic_uniform_buffers: u32,
    dynamic_storage_buffers: u32,
    sampled_textures: PerStageBindingTypeCounter,
    samplers: PerStageBindingTypeCounter,
    storage_buffers: PerStageBindingTypeCounter,
    storage_textures: PerStageBindingTypeCounter,
    uniform_buffers: PerStageBindingTypeCounter,
}

impl BindingTypeMaxCountValidator {
    pub(crate) fn add_binding(&mut self, binding: &wgt::BindGroupLayoutEntry) {
        let count = binding.count.map_or(1, |count| count.get());
        match binding.ty {
            wgt::BindingType::Buffer {
                ty: wgt::BufferBindingType::Uniform,
                has_dynamic_offset,
                ..
            } => {
                self.uniform_buffers.add(binding.visibility, count);
                if has_dynamic_offset {
                    self.dynamic_uniform_buffers += count;
                }
            }
            wgt::BindingType::Buffer {
                ty: wgt::BufferBindingType::Storage { .. },
                has_dynamic_offset,
                ..
            } => {
                self.storage_buffers.add(binding.visibility, count);
                if has_dynamic_offset {
                    self.dynamic_storage_buffers += count;
                }
            }
            wgt::BindingType::Sampler { .. } => {
                self.samplers.add(binding.visibility, count);
            }
            wgt::BindingType::Texture { .. } => {
                self.sampled_textures.add(binding.visibility, count);
            }
            wgt::BindingType::StorageTexture { .. } => {
                self.storage_textures.add(binding.visibility, count);
            }
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.dynamic_uniform_buffers += other.dynamic_uniform_buffers;
        self.dynamic_storage_buffers += other.dynamic_storage_buffers;
        self.sampled_textures.merge(&other.sampled_textures);
        self.samplers.merge(&other.samplers);
        self.storage_buffers.merge(&other.storage_buffers);
        self.storage_textures.merge(&other.storage_textures);
        self.uniform_buffers.merge(&other.uniform_buffers);
    }

    pub(crate) fn validate(&self, limits: &wgt::Limits) -> Result<(), BindingTypeMaxCountError> {
        if limits.max_dynamic_uniform_buffers_per_pipeline_layout < self.dynamic_uniform_buffers {
            return Err(BindingTypeMaxCountError {
                kind: BindingTypeMaxCountErrorKind::DynamicUniformBuffers,
                zone: BindingZone::Pipeline,
                limit: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
                count: self.dynamic_uniform_buffers,
            });
        }
        if limits.max_dynamic_storage_buffers_per_pipeline_layout < self.dynamic_storage_buffers {
            return Err(BindingTypeMaxCountError {
                kind: BindingTypeMaxCountErrorKind::DynamicStorageBuffers,
                zone: BindingZone::Pipeline,
                limit: limits.max_dynamic_storage_buffers_per_pipeline_layout,
                count: self.dynamic_storage_buffers,
            });
        }
        self.sampled_textures.validate(
            limits.max_sampled_textures_per_shader_stage,
            BindingTypeMaxCountErrorKind::SampledTextures,
        )?;
        self.storage_buffers.validate(
            limits.max_storage_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageBuffers,
        )?;
        self.samplers.validate(
            limits.max_samplers_per_shader_stage,
            BindingTypeMaxCountErrorKind::Samplers,
        )?;
        self.storage_buffers.validate(
            limits.max_storage_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageBuffers,
        )?;
        self.storage_textures.validate(
            limits.max_storage_textures_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageTextures,
        )?;
        self.uniform_buffers.validate(
            limits.max_uniform_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::UniformBuffers,
        )?;
        Ok(())
    }
}

/// Bindable resource and the slot to bind it to.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupEntry<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource<'a>,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: BindGroupLayoutId,
    /// The resources to bind to this bind group.
    pub entries: Cow<'a, [BindGroupEntry<'a>]>,
}

/// Describes a [`BindGroupLayout`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Array of entries in this BindGroupLayout
    pub entries: Cow<'a, [wgt::BindGroupLayoutEntry]>,
}

pub(crate) type BindEntryMap = FastHashMap<u32, wgt::BindGroupLayoutEntry>;

/// Bind group layout.
///
/// The lifetime of BGLs is a bit special. They are only referenced on CPU
/// without considering GPU operations. And on CPU they get manual
/// inc-refs and dec-refs. In particular, the following objects depend on them:
///  - produced bind groups
///  - produced pipeline layouts
///  - pipelines with implicit layouts
#[derive(Debug)]
pub struct BindGroupLayout<A: hal::Api> {
    pub(crate) raw: A::BindGroupLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) multi_ref_count: MultiRefCount,
    pub(crate) entries: BindEntryMap,
    #[allow(unused)]
    pub(crate) dynamic_count: usize,
    pub(crate) count_validator: BindingTypeMaxCountValidator,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl<A: hal::Api> Resource for BindGroupLayout<A> {
    const TYPE: &'static str = "BindGroupLayout";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreatePipelineLayoutError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("bind group layout {0:?} is invalid")]
    InvalidBindGroupLayout(BindGroupLayoutId),
    #[error(
        "push constant at index {index} has range bound {bound} not aligned to {}",
        wgt::PUSH_CONSTANT_ALIGNMENT
    )]
    MisalignedPushConstantRange { index: usize, bound: u32 },
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("push constant range (index {index}) provides for stage(s) {provided:?} but there exists another range that provides stage(s) {intersected:?}. Each stage may only be provided by one range")]
    MoreThanOnePushConstantRangePerStage {
        index: usize,
        provided: wgt::ShaderStages,
        intersected: wgt::ShaderStages,
    },
    #[error("push constant at index {index} has range {}..{} which exceeds device push constant size limit 0..{max}", range.start, range.end)]
    PushConstantRangeTooLarge {
        index: usize,
        range: Range<u32>,
        max: u32,
    },
    #[error(transparent)]
    TooManyBindings(BindingTypeMaxCountError),
    #[error("bind group layout count {actual} exceeds device bind group limit {max}")]
    TooManyGroups { actual: usize, max: usize },
}

impl PrettyError for CreatePipelineLayoutError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        if let Self::InvalidBindGroupLayout(id) = *self {
            fmt.bind_group_layout_label(&id);
        };
    }
}

#[derive(Clone, Debug, Error)]
pub enum PushConstantUploadError {
    #[error("provided push constant with indices {offset}..{end_offset} overruns matching push constant range at index {idx}, with stage(s) {:?} and indices {:?}", range.stages, range.range)]
    TooLarge {
        offset: u32,
        end_offset: u32,
        idx: usize,
        range: wgt::PushConstantRange,
    },
    #[error("provided push constant is for stage(s) {actual:?}, stage with a partial match found at index {idx} with stage(s) {matched:?}, however push constants must be complete matches")]
    PartialRangeMatch {
        actual: wgt::ShaderStages,
        idx: usize,
        matched: wgt::ShaderStages,
    },
    #[error("provided push constant is for stage(s) {actual:?}, but intersects a push constant range (at index {idx}) with stage(s) {missing:?}. Push constants must provide the stages for all ranges they intersect")]
    MissingStages {
        actual: wgt::ShaderStages,
        idx: usize,
        missing: wgt::ShaderStages,
    },
    #[error("provided push constant is for stage(s) {actual:?}, however the pipeline layout has no push constant range for the stage(s) {unmatched:?}")]
    UnmatchedStages {
        actual: wgt::ShaderStages,
        unmatched: wgt::ShaderStages,
    },
    #[error("provided push constant offset {0} does not respect `PUSH_CONSTANT_ALIGNMENT`")]
    Unaligned(u32),
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be used to create a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeine layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: Cow<'a, [BindGroupLayoutId]>,
    /// Set of push constant ranges this pipeline uses. Each shader stage that uses push constants
    /// must define the range in push constant memory that corresponds to its single `layout(push_constant)`
    /// uniform block.
    ///
    /// If this array is non-empty, the [`Features::PUSH_CONSTANTS`](wgt::Features::PUSH_CONSTANTS) must be enabled.
    pub push_constant_ranges: Cow<'a, [wgt::PushConstantRange]>,
}

#[derive(Debug)]
pub struct PipelineLayout<A: hal::Api> {
    pub(crate) raw: A::PipelineLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) bind_group_layout_ids: ArrayVec<Valid<BindGroupLayoutId>, { hal::MAX_BIND_GROUPS }>,
    pub(crate) push_constant_ranges: ArrayVec<wgt::PushConstantRange, { SHADER_STAGE_COUNT }>,
}

impl<A: hal::Api> PipelineLayout<A> {
    /// Validate push constants match up with expected ranges.
    pub(crate) fn validate_push_constant_ranges(
        &self,
        stages: wgt::ShaderStages,
        offset: u32,
        end_offset: u32,
    ) -> Result<(), PushConstantUploadError> {
        // Don't need to validate size against the push constant size limit here,
        // as push constant ranges are already validated to be within bounds,
        // and we validate that they are within the ranges.

        if offset % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
            return Err(PushConstantUploadError::Unaligned(offset));
        }

        // Push constant validation looks very complicated on the surface, but
        // the problem can be range-reduced pretty well.
        //
        // Push constants require (summarized from the vulkan spec):
        // 1. For each byte in the range and for each shader stage in stageFlags,
        //    there must be a push constant range in the layout that includes that
        //    byte and that stage.
        // 2. For each byte in the range and for each push constant range that overlaps that byte,
        //    `stage` must include all stages in that push constant range’s `stage`.
        //
        // However there are some additional constraints that help us:
        // 3. All push constant ranges are the only range that can access that stage.
        //    i.e. if one range has VERTEX, no other range has VERTEX
        //
        // Therefore we can simplify the checks in the following ways:
        // - Because 3 guarantees that the push constant range has a unique stage,
        //   when we check for 1, we can simply check that our entire updated range
        //   is within a push constant range. i.e. our range for a specific stage cannot
        //   intersect more than one push constant range.
        let mut used_stages = wgt::ShaderStages::NONE;
        for (idx, range) in self.push_constant_ranges.iter().enumerate() {
            // contains not intersects due to 2
            if stages.contains(range.stages) {
                if !(range.range.start <= offset && end_offset <= range.range.end) {
                    return Err(PushConstantUploadError::TooLarge {
                        offset,
                        end_offset,
                        idx,
                        range: range.clone(),
                    });
                }
                used_stages |= range.stages;
            } else if stages.intersects(range.stages) {
                // Will be caught by used stages check below, but we can do this because of 1
                // and is more helpful to the user.
                return Err(PushConstantUploadError::PartialRangeMatch {
                    actual: stages,
                    idx,
                    matched: range.stages,
                });
            }

            // The push constant range intersects range we are uploading
            if offset < range.range.end && range.range.start < end_offset {
                // But requires stages we don't provide
                if !stages.contains(range.stages) {
                    return Err(PushConstantUploadError::MissingStages {
                        actual: stages,
                        idx,
                        missing: stages,
                    });
                }
            }
        }
        if used_stages != stages {
            return Err(PushConstantUploadError::UnmatchedStages {
                actual: stages,
                unmatched: stages - used_stages,
            });
        }
        Ok(())
    }
}

impl<A: hal::Api> Resource for PipelineLayout<A> {
    const TYPE: &'static str = "PipelineLayout";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[repr(C)]
#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BufferBinding {
    pub buffer_id: BufferId,
    pub offset: wgt::BufferAddress,
    pub size: Option<wgt::BufferSize>,
}

// Note: Duplicated in `wgpu-rs` as `BindingResource`
// They're different enough that it doesn't make sense to share a common type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum BindingResource<'a> {
    Buffer(BufferBinding),
    BufferArray(Cow<'a, [BufferBinding]>),
    Sampler(SamplerId),
    SamplerArray(Cow<'a, [SamplerId]>),
    TextureView(TextureViewId),
    TextureViewArray(Cow<'a, [TextureViewId]>),
}

#[derive(Clone, Debug, Error)]
pub enum BindError {
    #[error("number of dynamic offsets ({actual}) doesn't match the number of dynamic bindings in the bind group layout ({expected})")]
    MismatchedDynamicOffsetCount { actual: usize, expected: usize },
    #[error(
        "dynamic binding at index {idx}: offset {offset} does not respect device's requested `{limit_name}` limit {alignment}"
    )]
    UnalignedDynamicBinding {
        idx: usize,
        offset: u32,
        alignment: u32,
        limit_name: &'static str,
    },
    #[error("dynamic binding at index {idx} with offset {offset} would overrun the buffer (limit: {max})")]
    DynamicBindingOutOfBounds { idx: usize, offset: u32, max: u64 },
}

#[derive(Debug)]
pub struct BindGroupDynamicBindingData {
    /// The maximum value the dynamic offset can have before running off the end of the buffer.
    pub(crate) maximum_dynamic_offset: wgt::BufferAddress,
    /// The binding type.
    pub(crate) binding_type: wgt::BufferBindingType,
}

pub(crate) fn buffer_binding_type_alignment(
    limits: &wgt::Limits,
    binding_type: wgt::BufferBindingType,
) -> (u32, &'static str) {
    match binding_type {
        wgt::BufferBindingType::Uniform => (
            limits.min_uniform_buffer_offset_alignment,
            "min_uniform_buffer_offset_alignment",
        ),
        wgt::BufferBindingType::Storage { .. } => (
            limits.min_storage_buffer_offset_alignment,
            "min_storage_buffer_offset_alignment",
        ),
    }
}

pub struct BindGroup<A: HalApi> {
    pub(crate) raw: A::BindGroup,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) layout_id: Valid<BindGroupLayoutId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) used: BindGroupStates<A>,
    pub(crate) used_buffer_ranges: Vec<BufferInitTrackerAction>,
    pub(crate) used_texture_ranges: Vec<TextureInitTrackerAction>,
    pub(crate) dynamic_binding_info: Vec<BindGroupDynamicBindingData>,
    /// Actual binding sizes for buffers that don't have `min_binding_size`
    /// specified in BGL. Listed in the order of iteration of `BGL.entries`.
    pub(crate) late_buffer_binding_sizes: Vec<wgt::BufferSize>,
}

impl<A: HalApi> BindGroup<A> {
    pub(crate) fn validate_dynamic_bindings(
        &self,
        offsets: &[wgt::DynamicOffset],
        limits: &wgt::Limits,
    ) -> Result<(), BindError> {
        if self.dynamic_binding_info.len() != offsets.len() {
            return Err(BindError::MismatchedDynamicOffsetCount {
                expected: self.dynamic_binding_info.len(),
                actual: offsets.len(),
            });
        }

        for (idx, (info, &offset)) in self
            .dynamic_binding_info
            .iter()
            .zip(offsets.iter())
            .enumerate()
        {
            let (alignment, limit_name) = buffer_binding_type_alignment(limits, info.binding_type);
            if offset as wgt::BufferAddress % alignment as u64 != 0 {
                return Err(BindError::UnalignedDynamicBinding {
                    idx,
                    offset,
                    alignment,
                    limit_name,
                });
            }

            if offset as wgt::BufferAddress > info.maximum_dynamic_offset {
                return Err(BindError::DynamicBindingOutOfBounds {
                    idx,
                    offset,
                    max: info.maximum_dynamic_offset,
                });
            }
        }

        Ok(())
    }
}

impl<A: HalApi> Resource for BindGroup<A> {
    const TYPE: &'static str = "BindGroup";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
pub enum GetBindGroupLayoutError {
    #[error("pipeline is invalid")]
    InvalidPipeline,
    #[error("invalid group index {0}")]
    InvalidGroupIndex(u32),
}

#[derive(Clone, Debug, Error, PartialEq)]
#[error("Buffer is bound with size {bound_size} where the shader expects {shader_size} in group[{group_index}] compact index {compact_index}")]
pub struct LateMinBufferBindingSizeMismatch {
    pub group_index: u32,
    pub compact_index: usize,
    pub shader_size: wgt::BufferAddress,
    pub bound_size: wgt::BufferAddress,
}
