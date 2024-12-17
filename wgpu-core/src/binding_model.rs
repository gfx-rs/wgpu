use crate::{
    device::{
        bgl, Device, DeviceError, MissingDownlevelFlags, MissingFeatures, SHADER_STAGE_COUNT,
    },
    id::{BindGroupLayoutId, BufferId, SamplerId, TextureViewId, TlasId},
    init_tracker::{BufferInitTrackerAction, TextureInitTrackerAction},
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{
        Buffer, DestroyedResourceError, InvalidResourceError, Labeled, MissingBufferUsageError,
        MissingTextureUsageError, ResourceErrorIdent, Sampler, TextureView, TrackingData,
    },
    resource_log,
    snatch::{SnatchGuard, Snatchable},
    track::{BindGroupStates, ResourceUsageCompatibilityError},
    Label,
};

use arrayvec::ArrayVec;

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use std::{
    borrow::Cow,
    mem::ManuallyDrop,
    ops::Range,
    sync::{Arc, OnceLock, Weak},
};

use crate::resource::Tlas;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BindGroupLayoutEntryError {
    #[error("Cube dimension is not expected for texture storage")]
    StorageTextureCube,
    #[error("Read-write and read-only storage textures are not allowed by webgpu, they require the native only feature TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES")]
    StorageTextureReadWrite,
    #[error("Arrays of bindings unsupported for this type of binding")]
    ArrayUnsupported,
    #[error("Multisampled binding with sample type `TextureSampleType::Float` must have filterable set to false.")]
    SampleTypeFloatFilterableBindingMultisampled,
    #[error("Multisampled texture binding view dimension must be 2d, got {0:?}")]
    Non2DMultisampled(wgt::TextureViewDimension),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateBindGroupLayoutError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Conflicting binding at index {0}")]
    ConflictBinding(u32),
    #[error("Binding {binding} entry is invalid")]
    Entry {
        binding: u32,
        #[source]
        error: BindGroupLayoutEntryError,
    },
    #[error(transparent)]
    TooManyBindings(BindingTypeMaxCountError),
    #[error("Binding index {binding} is greater than the maximum number {maximum}")]
    InvalidBindingIndex { binding: u32, maximum: u32 },
    #[error("Invalid visibility {0:?}")]
    InvalidVisibility(wgt::ShaderStages),
}

//TODO: refactor this to move out `enum BindingError`.

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateBindGroupError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(
        "Binding count declared with at most {expected} items, but {actual} items were provided"
    )]
    BindingArrayPartialLengthMismatch { actual: usize, expected: usize },
    #[error(
        "Binding count declared with exactly {expected} items, but {actual} items were provided"
    )]
    BindingArrayLengthMismatch { actual: usize, expected: usize },
    #[error("Array binding provided zero elements")]
    BindingArrayZeroLength,
    #[error("The bound range {range:?} of {buffer} overflows its size ({size})")]
    BindingRangeTooLarge {
        buffer: ResourceErrorIdent,
        range: Range<wgt::BufferAddress>,
        size: u64,
    },
    #[error("Binding size {actual} of {buffer} is less than minimum {min}")]
    BindingSizeTooSmall {
        buffer: ResourceErrorIdent,
        actual: u64,
        min: u64,
    },
    #[error("{0} binding size is zero")]
    BindingZeroSize(ResourceErrorIdent),
    #[error("Number of bindings in bind group descriptor ({actual}) does not match the number of bindings defined in the bind group layout ({expected})")]
    BindingsNumMismatch { actual: usize, expected: usize },
    #[error("Binding {0} is used at least twice in the descriptor")]
    DuplicateBinding(u32),
    #[error("Unable to find a corresponding declaration for the given binding {0}")]
    MissingBindingDeclaration(u32),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error("Binding declared as a single item, but bind group is using it as an array")]
    SingleBindingExpected,
    #[error("Buffer offset {0} does not respect device's requested `{1}` limit {2}")]
    UnalignedBufferOffset(wgt::BufferAddress, &'static str, u32),
    #[error(
        "Buffer binding {binding} range {given} exceeds `max_*_buffer_binding_size` limit {limit}"
    )]
    BufferRangeTooLarge {
        binding: u32,
        given: u32,
        limit: u32,
    },
    #[error("Binding {binding} has a different type ({actual:?}) than the one in the layout ({expected:?})")]
    WrongBindingType {
        // Index of the binding
        binding: u32,
        // The type given to the function
        actual: wgt::BindingType,
        // Human-readable description of expected types
        expected: &'static str,
    },
    #[error("Texture binding {binding} expects multisampled = {layout_multisampled}, but given a view with samples = {view_samples}")]
    InvalidTextureMultisample {
        binding: u32,
        layout_multisampled: bool,
        view_samples: u32,
    },
    #[error(
        "Texture binding {} expects sample type {:?}, but was given a view with format {:?} (sample type {:?})",
        binding,
        layout_sample_type,
        view_format,
        view_sample_type
    )]
    InvalidTextureSampleType {
        binding: u32,
        layout_sample_type: wgt::TextureSampleType,
        view_format: wgt::TextureFormat,
        view_sample_type: wgt::TextureSampleType,
    },
    #[error("Texture binding {binding} expects dimension = {layout_dimension:?}, but given a view with dimension = {view_dimension:?}")]
    InvalidTextureDimension {
        binding: u32,
        layout_dimension: wgt::TextureViewDimension,
        view_dimension: wgt::TextureViewDimension,
    },
    #[error("Storage texture binding {binding} expects format = {layout_format:?}, but given a view with format = {view_format:?}")]
    InvalidStorageTextureFormat {
        binding: u32,
        layout_format: wgt::TextureFormat,
        view_format: wgt::TextureFormat,
    },
    #[error("Storage texture bindings must have a single mip level, but given a view with mip_level_count = {mip_level_count:?} at binding {binding}")]
    InvalidStorageTextureMipLevelCount { binding: u32, mip_level_count: u32 },
    #[error("Sampler binding {binding} expects comparison = {layout_cmp}, but given a sampler with comparison = {sampler_cmp}")]
    WrongSamplerComparison {
        binding: u32,
        layout_cmp: bool,
        sampler_cmp: bool,
    },
    #[error("Sampler binding {binding} expects filtering = {layout_flt}, but given a sampler with filtering = {sampler_flt}")]
    WrongSamplerFiltering {
        binding: u32,
        layout_flt: bool,
        sampler_flt: bool,
    },
    #[error("Bound texture views can not have both depth and stencil aspects enabled")]
    DepthStencilAspect,
    #[error("The adapter does not support read access for storage textures of format {0:?}")]
    StorageReadNotSupported(wgt::TextureFormat),
    #[error("The adapter does not support write access for storage textures of format {0:?}")]
    StorageWriteNotSupported(wgt::TextureFormat),
    #[error("The adapter does not support read-write access for storage textures of format {0:?}")]
    StorageReadWriteNotSupported(wgt::TextureFormat),
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error)]
pub enum BindingZone {
    #[error("Stage {0:?}")]
    Stage(wgt::ShaderStages),
    #[error("Whole pipeline")]
    Pipeline,
}

#[derive(Clone, Debug, Error)]
#[error("Too many bindings of type {kind:?} in {zone}, limit is {limit}, count was {count}. Check the limit `{}` passed to `Adapter::request_device`", .kind.to_config_str())]
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

impl BindingTypeMaxCountErrorKind {
    fn to_config_str(&self) -> &'static str {
        match self {
            BindingTypeMaxCountErrorKind::DynamicUniformBuffers => {
                "max_dynamic_uniform_buffers_per_pipeline_layout"
            }
            BindingTypeMaxCountErrorKind::DynamicStorageBuffers => {
                "max_dynamic_storage_buffers_per_pipeline_layout"
            }
            BindingTypeMaxCountErrorKind::SampledTextures => {
                "max_sampled_textures_per_shader_stage"
            }
            BindingTypeMaxCountErrorKind::Samplers => "max_samplers_per_shader_stage",
            BindingTypeMaxCountErrorKind::StorageBuffers => "max_storage_buffers_per_shader_stage",
            BindingTypeMaxCountErrorKind::StorageTextures => {
                "max_storage_textures_per_shader_stage"
            }
            BindingTypeMaxCountErrorKind::UniformBuffers => "max_uniform_buffers_per_shader_stage",
        }
    }
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
    acceleration_structures: PerStageBindingTypeCounter,
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
            wgt::BindingType::AccelerationStructure => {
                self.acceleration_structures.add(binding.visibility, count);
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BindGroupEntry<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource<'a>,
}

/// Bindable resource and the slot to bind it to.
#[derive(Clone, Debug)]
pub struct ResolvedBindGroupEntry<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: ResolvedBindingResource<'a>,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: BindGroupLayoutId,
    /// The resources to bind to this bind group.
    pub entries: Cow<'a, [BindGroupEntry<'a>]>,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone, Debug)]
pub struct ResolvedBindGroupDescriptor<'a> {
    /// Debug label of the bind group.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: Arc<BindGroupLayout>,
    /// The resources to bind to this bind group.
    pub entries: Cow<'a, [ResolvedBindGroupEntry<'a>]>,
}

/// Describes a [`BindGroupLayout`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Array of entries in this BindGroupLayout
    pub entries: Cow<'a, [wgt::BindGroupLayoutEntry]>,
}

/// Used by [`BindGroupLayout`]. It indicates whether the BGL must be
/// used with a specific pipeline. This constraint only happens when
/// the BGLs have been derived from a pipeline without a layout.
#[derive(Debug)]
pub(crate) enum ExclusivePipeline {
    None,
    Render(Weak<RenderPipeline>),
    Compute(Weak<ComputePipeline>),
}

impl std::fmt::Display for ExclusivePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExclusivePipeline::None => f.write_str("None"),
            ExclusivePipeline::Render(p) => {
                if let Some(p) = p.upgrade() {
                    p.error_ident().fmt(f)
                } else {
                    f.write_str("RenderPipeline")
                }
            }
            ExclusivePipeline::Compute(p) => {
                if let Some(p) = p.upgrade() {
                    p.error_ident().fmt(f)
                } else {
                    f.write_str("ComputePipeline")
                }
            }
        }
    }
}

/// Bind group layout.
#[derive(Debug)]
pub struct BindGroupLayout {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynBindGroupLayout>>,
    pub(crate) device: Arc<Device>,
    pub(crate) entries: bgl::EntryMap,
    /// It is very important that we know if the bind group comes from the BGL pool.
    ///
    /// If it does, then we need to remove it from the pool when we drop it.
    ///
    /// We cannot unconditionally remove from the pool, as BGLs that don't come from the pool
    /// (derived BGLs) must not be removed.
    pub(crate) origin: bgl::Origin,
    pub(crate) exclusive_pipeline: OnceLock<ExclusivePipeline>,
    #[allow(unused)]
    pub(crate) binding_count_validator: BindingTypeMaxCountValidator,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
}

impl Drop for BindGroupLayout {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        if matches!(self.origin, bgl::Origin::Pool) {
            self.device.bgl_pool.remove(&self.entries);
        }
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_bind_group_layout(raw);
        }
    }
}

crate::impl_resource_type!(BindGroupLayout);
crate::impl_labeled!(BindGroupLayout);
crate::impl_parent_device!(BindGroupLayout);
crate::impl_storage_item!(BindGroupLayout);

impl BindGroupLayout {
    pub(crate) fn raw(&self) -> &dyn hal::DynBindGroupLayout {
        self.raw.as_ref()
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreatePipelineLayoutError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(
        "Push constant at index {index} has range bound {bound} not aligned to {}",
        wgt::PUSH_CONSTANT_ALIGNMENT
    )]
    MisalignedPushConstantRange { index: usize, bound: u32 },
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("Push constant range (index {index}) provides for stage(s) {provided:?} but there exists another range that provides stage(s) {intersected:?}. Each stage may only be provided by one range")]
    MoreThanOnePushConstantRangePerStage {
        index: usize,
        provided: wgt::ShaderStages,
        intersected: wgt::ShaderStages,
    },
    #[error("Push constant at index {index} has range {}..{} which exceeds device push constant size limit 0..{max}", range.start, range.end)]
    PushConstantRangeTooLarge {
        index: usize,
        range: Range<u32>,
        max: u32,
    },
    #[error(transparent)]
    TooManyBindings(BindingTypeMaxCountError),
    #[error("Bind group layout count {actual} exceeds device bind group limit {max}")]
    TooManyGroups { actual: usize, max: usize },
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum PushConstantUploadError {
    #[error("Provided push constant with indices {offset}..{end_offset} overruns matching push constant range at index {idx}, with stage(s) {:?} and indices {:?}", range.stages, range.range)]
    TooLarge {
        offset: u32,
        end_offset: u32,
        idx: usize,
        range: wgt::PushConstantRange,
    },
    #[error("Provided push constant is for stage(s) {actual:?}, stage with a partial match found at index {idx} with stage(s) {matched:?}, however push constants must be complete matches")]
    PartialRangeMatch {
        actual: wgt::ShaderStages,
        idx: usize,
        matched: wgt::ShaderStages,
    },
    #[error("Provided push constant is for stage(s) {actual:?}, but intersects a push constant range (at index {idx}) with stage(s) {missing:?}. Push constants must provide the stages for all ranges they intersect")]
    MissingStages {
        actual: wgt::ShaderStages,
        idx: usize,
        missing: wgt::ShaderStages,
    },
    #[error("Provided push constant is for stage(s) {actual:?}, however the pipeline layout has no push constant range for the stage(s) {unmatched:?}")]
    UnmatchedStages {
        actual: wgt::ShaderStages,
        unmatched: wgt::ShaderStages,
    },
    #[error("Provided push constant offset {0} does not respect `PUSH_CONSTANT_ALIGNMENT`")]
    Unaligned(u32),
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be used to create a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeline layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: Cow<'a, [BindGroupLayoutId]>,
    /// Set of push constant ranges this pipeline uses. Each shader stage that
    /// uses push constants must define the range in push constant memory that
    /// corresponds to its single `layout(push_constant)` uniform block.
    ///
    /// If this array is non-empty, the
    /// [`Features::PUSH_CONSTANTS`](wgt::Features::PUSH_CONSTANTS) feature must
    /// be enabled.
    pub push_constant_ranges: Cow<'a, [wgt::PushConstantRange]>,
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be used to create a pipeline layout.
#[derive(Debug)]
pub struct ResolvedPipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeline layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: Cow<'a, [Arc<BindGroupLayout>]>,
    /// Set of push constant ranges this pipeline uses. Each shader stage that
    /// uses push constants must define the range in push constant memory that
    /// corresponds to its single `layout(push_constant)` uniform block.
    ///
    /// If this array is non-empty, the
    /// [`Features::PUSH_CONSTANTS`](wgt::Features::PUSH_CONSTANTS) feature must
    /// be enabled.
    pub push_constant_ranges: Cow<'a, [wgt::PushConstantRange]>,
}

#[derive(Debug)]
pub struct PipelineLayout {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynPipelineLayout>>,
    pub(crate) device: Arc<Device>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) bind_group_layouts: ArrayVec<Arc<BindGroupLayout>, { hal::MAX_BIND_GROUPS }>,
    pub(crate) push_constant_ranges: ArrayVec<wgt::PushConstantRange, { SHADER_STAGE_COUNT }>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_pipeline_layout(raw);
        }
    }
}

impl PipelineLayout {
    pub(crate) fn raw(&self) -> &dyn hal::DynPipelineLayout {
        self.raw.as_ref()
    }

    pub(crate) fn get_binding_maps(&self) -> ArrayVec<&bgl::EntryMap, { hal::MAX_BIND_GROUPS }> {
        self.bind_group_layouts
            .iter()
            .map(|bgl| &bgl.entries)
            .collect()
    }

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

crate::impl_resource_type!(PipelineLayout);
crate::impl_labeled!(PipelineLayout);
crate::impl_parent_device!(PipelineLayout);
crate::impl_storage_item!(PipelineLayout);

#[repr(C)]
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BufferBinding {
    pub buffer_id: BufferId,
    pub offset: wgt::BufferAddress,
    pub size: Option<wgt::BufferSize>,
}

#[derive(Clone, Debug)]
pub struct ResolvedBufferBinding {
    pub buffer: Arc<Buffer>,
    pub offset: wgt::BufferAddress,
    pub size: Option<wgt::BufferSize>,
}

// Note: Duplicated in `wgpu-rs` as `BindingResource`
// They're different enough that it doesn't make sense to share a common type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BindingResource<'a> {
    Buffer(BufferBinding),
    BufferArray(Cow<'a, [BufferBinding]>),
    Sampler(SamplerId),
    SamplerArray(Cow<'a, [SamplerId]>),
    TextureView(TextureViewId),
    TextureViewArray(Cow<'a, [TextureViewId]>),
    AccelerationStructure(TlasId),
}

// Note: Duplicated in `wgpu-rs` as `BindingResource`
// They're different enough that it doesn't make sense to share a common type
#[derive(Debug, Clone)]
pub enum ResolvedBindingResource<'a> {
    Buffer(ResolvedBufferBinding),
    BufferArray(Cow<'a, [ResolvedBufferBinding]>),
    Sampler(Arc<Sampler>),
    SamplerArray(Cow<'a, [Arc<Sampler>]>),
    TextureView(Arc<TextureView>),
    TextureViewArray(Cow<'a, [Arc<TextureView>]>),
    AccelerationStructure(Arc<Tlas>),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BindError {
    #[error(
        "{bind_group} {group} expects {expected} dynamic offset{s0}. However {actual} dynamic offset{s1} were provided.",
        s0 = if *.expected >= 2 { "s" } else { "" },
        s1 = if *.actual >= 2 { "s" } else { "" },
    )]
    MismatchedDynamicOffsetCount {
        bind_group: ResourceErrorIdent,
        group: u32,
        actual: usize,
        expected: usize,
    },
    #[error(
        "Dynamic binding index {idx} (targeting {bind_group} {group}, binding {binding}) with value {offset}, does not respect device's requested `{limit_name}` limit: {alignment}"
    )]
    UnalignedDynamicBinding {
        bind_group: ResourceErrorIdent,
        idx: usize,
        group: u32,
        binding: u32,
        offset: u32,
        alignment: u32,
        limit_name: &'static str,
    },
    #[error(
        "Dynamic binding offset index {idx} with offset {offset} would overrun the buffer bound to {bind_group} {group} -> binding {binding}. \
         Buffer size is {buffer_size} bytes, the binding binds bytes {binding_range:?}, meaning the maximum the binding can be offset is {maximum_dynamic_offset} bytes",
    )]
    DynamicBindingOutOfBounds {
        bind_group: ResourceErrorIdent,
        idx: usize,
        group: u32,
        binding: u32,
        offset: u32,
        buffer_size: wgt::BufferAddress,
        binding_range: Range<wgt::BufferAddress>,
        maximum_dynamic_offset: wgt::BufferAddress,
    },
}

#[derive(Debug)]
pub struct BindGroupDynamicBindingData {
    /// The index of the binding.
    ///
    /// Used for more descriptive errors.
    pub(crate) binding_idx: u32,
    /// The size of the buffer.
    ///
    /// Used for more descriptive errors.
    pub(crate) buffer_size: wgt::BufferAddress,
    /// The range that the binding covers.
    ///
    /// Used for more descriptive errors.
    pub(crate) binding_range: Range<wgt::BufferAddress>,
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

pub(crate) fn buffer_binding_type_bounds_check_alignment(
    alignments: &hal::Alignments,
    binding_type: wgt::BufferBindingType,
) -> wgt::BufferAddress {
    match binding_type {
        wgt::BufferBindingType::Uniform => alignments.uniform_bounds_check_alignment.get(),
        wgt::BufferBindingType::Storage { .. } => wgt::COPY_BUFFER_ALIGNMENT,
    }
}

#[derive(Debug)]
pub struct BindGroup {
    pub(crate) raw: Snatchable<Box<dyn hal::DynBindGroup>>,
    pub(crate) device: Arc<Device>,
    pub(crate) layout: Arc<BindGroupLayout>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    pub(crate) used: BindGroupStates,
    pub(crate) used_buffer_ranges: Vec<BufferInitTrackerAction>,
    pub(crate) used_texture_ranges: Vec<TextureInitTrackerAction>,
    pub(crate) dynamic_binding_info: Vec<BindGroupDynamicBindingData>,
    /// Actual binding sizes for buffers that don't have `min_binding_size`
    /// specified in BGL. Listed in the order of iteration of `BGL.entries`.
    pub(crate) late_buffer_binding_sizes: Vec<wgt::BufferSize>,
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                self.device.raw().destroy_bind_group(raw);
            }
        }
    }
}

impl BindGroup {
    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a dyn hal::DynBindGroup, DestroyedResourceError> {
        // Clippy insist on writing it this way. The idea is to return None
        // if any of the raw buffer is not valid anymore.
        for buffer in &self.used_buffer_ranges {
            buffer.buffer.try_raw(guard)?;
        }
        for texture in &self.used_texture_ranges {
            texture.texture.try_raw(guard)?;
        }

        self.raw
            .get(guard)
            .map(|raw| raw.as_ref())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn validate_dynamic_bindings(
        &self,
        bind_group_index: u32,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), BindError> {
        if self.dynamic_binding_info.len() != offsets.len() {
            return Err(BindError::MismatchedDynamicOffsetCount {
                bind_group: self.error_ident(),
                group: bind_group_index,
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
            let (alignment, limit_name) =
                buffer_binding_type_alignment(&self.device.limits, info.binding_type);
            if offset as wgt::BufferAddress % alignment as u64 != 0 {
                return Err(BindError::UnalignedDynamicBinding {
                    bind_group: self.error_ident(),
                    group: bind_group_index,
                    binding: info.binding_idx,
                    idx,
                    offset,
                    alignment,
                    limit_name,
                });
            }

            if offset as wgt::BufferAddress > info.maximum_dynamic_offset {
                return Err(BindError::DynamicBindingOutOfBounds {
                    bind_group: self.error_ident(),
                    group: bind_group_index,
                    binding: info.binding_idx,
                    idx,
                    offset,
                    buffer_size: info.buffer_size,
                    binding_range: info.binding_range.clone(),
                    maximum_dynamic_offset: info.maximum_dynamic_offset,
                });
            }
        }

        Ok(())
    }
}

crate::impl_resource_type!(BindGroup);
crate::impl_labeled!(BindGroup);
crate::impl_parent_device!(BindGroup);
crate::impl_storage_item!(BindGroup);
crate::impl_trackable!(BindGroup);

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum GetBindGroupLayoutError {
    #[error("Invalid group index {0}")]
    InvalidGroupIndex(u32),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("Buffer is bound with size {bound_size} where the shader expects {shader_size} in group[{group_index}] compact index {compact_index}")]
pub struct LateMinBufferBindingSizeMismatch {
    pub group_index: u32,
    pub compact_index: usize,
    pub shader_size: wgt::BufferAddress,
    pub bound_size: wgt::BufferAddress,
}
