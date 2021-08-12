use crate::{
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures, SHADER_STAGE_COUNT},
    error::{ErrorFormatter, PrettyError},
    hub::Resource,
    id::{self, AllResources, AnyBackend, BufferId, Hkt, TextureViewId},
    memory_init_tracker::MemoryInitTrackerAction,
    track::{TrackerSet, UsageConflict},
    validation::{MissingBufferUsageError, MissingTextureUsageError},
    FastHashMap, Label, LifeGuard,
};
#[cfg(feature = "trace")]
use crate::{
    command::FromCommand,
    id::BorrowHkt,
};

use arrayvec::ArrayVec;

#[cfg(feature = "replay")]
use serde::Deserialize;
#[cfg(feature = "trace")]
use serde::Serialize;

#[cfg(feature = "replay")] use core::convert::{TryFrom, TryInto};
use std::{
    borrow::Cow,
    mem::ManuallyDrop,
    ops::Range,
};

use hal::{Device as _};
use thiserror::Error;

#[derive(Clone, Debug, Error)]
pub enum BindGroupLayoutEntryError {
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
    #[error("sampler binding at index {0:?} is invalid")]
    InvalidSampler(/*SamplerId*/u32),
    #[error("binding count declared with {expected} items, but {actual} items were provided")]
    BindingArrayLengthMismatch { actual: usize, expected: usize },
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
    #[error("buffer offset {0} does not respect `BIND_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(wgt::BufferAddress),
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
        match self {
            Self::BindingZeroSize(id) => {
                fmt.buffer_label(id);
            }
            Self::BindingRangeTooLarge { buffer, .. } => {
                fmt.buffer_label(buffer);
            }
            Self::BindingSizeTooSmall { buffer, .. } => {
                fmt.buffer_label(buffer);
            }
            Self::InvalidBuffer(id) => {
                fmt.buffer_label(id);
            }
            Self::InvalidTextureView(id) => {
                fmt.texture_view_label(id);
            }
            Self::InvalidSampler(_id) => {
                // TODO: Figure out what to do here?
                // fmt.sampler_label(id);
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
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupEntry<'a, A: hal::Api, F: AllResources<A>> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "BindingResource<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "BindingResource<'a, A, F>: serde::Deserialize<'de>")))]
    pub resource: BindingResource<'a, A, F>,
}
pub type BindGroupEntryIn<'a, A> = BindGroupEntry<'a, A, id::IdGuardCon<'a>>;

impl<'a, A: hal::Api, F: AllResources<A>> BindGroupEntry<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(&'b self, f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>) -> Result<(), E> {
        self.resource.trace_resources(f)
    }
}

#[cfg(feature = "trace")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<BindGroupEntry<'a, A, F>> for BindGroupEntry<'b, B, G>
    where
        BindingResource<'b, B, G>:
            FromCommand<BindingResource<'a, A, F>>,
{
    fn from(desc: BindGroupEntry<'a, A, F>) -> Self {
        Self {
            binding: desc.binding,
            resource: FromCommand::from(desc.resource),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api + 'b, B: hal::Api, F: AllResources<A> + 'b, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, &'b BindGroupEntry<'a, A, F>)> for BindGroupEntry<'b, B, G>
    where
        /*(&'b id::IdCache2, &'b <F as Hkt<crate::resource::Sampler<A>>>::Output):
            TryInto<<G as Hkt<crate::resource::Sampler<B>>>::Output, Error=E>,*/
        (&'b id::IdCache2, &'b BindingResource<'a, A, F>):
            TryInto<BindingResource<'b, B, G>, Error=E>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, &'b BindGroupEntry<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            binding: desc.binding,
            resource: (cache, &desc.resource).try_into()?,
        })
    }
}

/// Describes a group of bindings and the resources to be bound.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupDescriptor<'a, A: hal::Api, F: AllResources<A>, I> {
    /// Debug label of the bind group. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F::Owned as Hkt<BindGroupLayout<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F::Owned as Hkt<BindGroupLayout<A>>>::Output: serde::Deserialize<'de>")))]
    pub layout: <F::Owned as Hkt<BindGroupLayout<A>>>::Output,
    /// The resources to bind to this bind group.
    /* #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "I: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "I: serde::Deserialize<'de>")))] */
    pub entries: /*Vec<BindGroupEntry<'a, A, F>>*/I,
}

pub type BindGroupDescriptorIn<'a, A, I> = BindGroupDescriptor<'a, A, id::IdGuardCon<'a>, I>;

impl<'a, A: hal::Api, F: AllResources<A>, I> BindGroupDescriptor<'a, A, F, I> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where
            id::Cached<A, &'b F>: 'b,
            <&'b F::Owned as Hkt<BindGroupLayout<A>>>::Output:
                Into<&'b <F as Hkt<BindGroupLayout<A>>>::Output>,
            &'b I: IntoIterator<Item=&'b BindGroupEntry<'a, A, F>>,
    {
        f(BindGroupLayout::upcast((&self.layout).into()))?;
        self.entries.into_iter().try_for_each(|entry| entry.trace_resources(&mut f))
    }
}

/* #[cfg(feature = "trace")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<BindGroupDescriptor<'a, A, F>> for BindGroupDescriptor<'b, B, G>
    where
        I: IntoIterator<Item=&'b BindGroupEntry<'a, A, F>>,
        ProgrammableStageDescriptor<'a, B, G>:
            FromCommand<&'b ProgrammableStageDescriptor<'a, A, F>>,
        A: crate::hub::HalApi,
        G::Owned: BorrowHkt<A, BindGroupLayout<B>, BindGroupLayout<hal::api::Empty>, F::Owned>,

        <F as Hkt<crate::resource::Sampler<A>>>::Output:
            Into<<G as Hkt<crate::resource::Sampler<B>>>::Output>,
{
    fn from(desc: BindingResource<'a, A, F>) -> Self {
        use BindingResource::*;

        match desc {
            Buffer(bb) => Buffer(bb),
            BufferArray(bindings_array) => BufferArray(bindings_array),
            Sampler(id) => Sampler(id.into()),
            TextureView(id) => TextureView(id),
            TextureViewArray(bindings_array) => TextureViewArray(bindings_array),
        }
    }
} */

/// Describes a [`BindGroupLayout`].
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Array of entries in this BindGroupLayout
    pub entries: Cow<'a, [wgt::BindGroupLayoutEntry]>,
}

impl<'a> BindGroupLayoutDescriptor<'a> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, A: hal::Api, F: AllResources<A>, E>(
        &'b self,
        _f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        // Nothing to trace.
        Ok(())
    }
}

pub(crate) type BindEntryMap = FastHashMap<u32, wgt::BindGroupLayoutEntry>;

#[derive(Debug)]
pub struct BindGroupLayout<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::BindGroupLayout>,
    pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    // pub(crate) multi_ref_count: MultiRefCount,
    /// Invariant: no duplicates, sorted in ascending order by entry key.
    pub(crate) entries: Box<[wgt::BindGroupLayoutEntry]>,
    pub(crate) dynamic_count: usize,
    pub(crate) count_validator: BindingTypeMaxCountValidator,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl<A: hal::Api> BindGroupLayout<A> {
    /// Used to help with implementing bind group deduplication outside of wgpu-core.
    pub fn entries(&self) -> &[wgt::BindGroupLayoutEntry] {
        &self.entries
    }

    /// Used to help with implementing bind group deduplication outside of wgpu-core.
    pub fn device_id(&self) -> id::IdGuard<A, Device<id::Dummy>> where A: crate::hub::HalApi {
        self.device_id.borrow()
    }
}

impl<A: hal::Api> Resource for BindGroupLayout<A> {
    const TYPE: &'static str = "BindGroupLayout";

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        _: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        _: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: crate::hub::HalApi + 'b,
    {
        // Nothing to trace.
        Ok(())
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // unreachable!()
    }

    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

impl<A: hal::Api> Drop for BindGroupLayout<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the bind group layout is uniquely owned, so it is unused by any CPU
                // resources, and bind group layouts do not directly hold onto GPU resources, so
                // calling destroy_bind_group_layout is safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.device_id.raw.destroy_bind_group_layout(ManuallyDrop::take(&mut self.raw));
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreatePipelineLayoutError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("bind group layout at index {0:?} is invalid")]
    InvalidBindGroupLayout(usize),
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
        if let Self::InvalidBindGroupLayout(_id) = self {
            // TODO: Figure out what to do here?
            // fmt.bind_group_layout_label(&id);
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
#[derive(/*Clone, */Debug/*, PartialEq, Eq, Hash*/)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PipelineLayoutDescriptor<'a, A: hal::Api, F: AllResources<A>> {
    /// Debug label of the pipeine layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F::Owned as Hkt<BindGroupLayout<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F::Owned as Hkt<BindGroupLayout<A>>>::Output: serde::Deserialize<'de>")))]
    pub bind_group_layouts: /*Cow<'a, [BindGroupLayoutId]>*/ArrayVec<<F::Owned as Hkt<BindGroupLayout<A>>>::Output, { hal::MAX_BIND_GROUPS }>,
    /// Set of push constant ranges this pipeline uses. Each shader stage that uses push constants
    /// must define the range in push constant memory that corresponds to its single `layout(push_constant)`
    /// uniform block.
    ///
    /// If this array is non-empty, the [`Features::PUSH_CONSTANTS`](wgt::Features::PUSH_CONSTANTS) must be enabled.
    pub push_constant_ranges: Cow<'a, [wgt::PushConstantRange]>,
}

pub type PipelineLayoutDescriptorIn<'a, A> = PipelineLayoutDescriptor<'a, A, id::IdGuardCon<'a>>;

impl<'a, A: hal::Api, F: AllResources<A>> PipelineLayoutDescriptor<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where
            id::Cached<A, &'b F>: 'b,
            <&'b F::Owned as Hkt<BindGroupLayout<A>>>::Output:
                Into<&'b <F as Hkt<BindGroupLayout<A>>>::Output>,
    {
        self.bind_group_layouts.iter()
            .try_for_each(|bind_group_layout| f(BindGroupLayout::upcast(bind_group_layout.into())))
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b PipelineLayoutDescriptor<'a, A, F>> for PipelineLayoutDescriptor<'b, B, G>
    where
        A: crate::hub::HalApi,
        G::Owned: BorrowHkt<A, BindGroupLayout<B>, BindGroupLayout<hal::api::Empty>, F::Owned>,
{
    fn from(desc: &'b PipelineLayoutDescriptor<'a, A, F>) -> Self {
        Self {
            label: desc.label.as_deref().map(Cow::Borrowed),
            bind_group_layouts: desc.bind_group_layouts.iter().map(G::Owned::borrow).collect(),
            push_constant_ranges: Cow::Borrowed(&*desc.push_constant_ranges),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'a id::IdCache2, PipelineLayoutDescriptor<'a, A, F>)> for PipelineLayoutDescriptor<'a, B, G>
    where
        (&'a id::IdCache2, <F::Owned as Hkt<BindGroupLayout<A>>>::Output):
            TryInto<<G as Hkt<BindGroupLayout<B>>>::Output, Error=E>,
        <G as Hkt<BindGroupLayout<B>>>::Output:
            Into<<G::Owned as Hkt<BindGroupLayout<B>>>::Output>,
{
    type Error = E;

    fn try_from((cache, desc): (&'a id::IdCache2, PipelineLayoutDescriptor<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            label: desc.label,
            bind_group_layouts: desc.bind_group_layouts.into_iter()
                .map(|layout| Ok(TryInto::<<G as Hkt<BindGroupLayout<B>>>::Output>::try_into((cache, layout))?.into()))
                .collect::<Result<_, _>>()?,
            push_constant_ranges: desc.push_constant_ranges,
        })
    }
}


#[derive(Debug)]
pub struct PipelineLayout<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::PipelineLayout>,
    pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    // pub(crate) life_guard: LifeGuard,
    pub(crate) bind_group_layout_ids: ArrayVec</*Valid<BindGroupLayoutId>*/id::ValidId2<BindGroupLayout<A>>, { hal::MAX_BIND_GROUPS }>,
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

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        id: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        mut f: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: crate::hub::HalApi + 'b,
    {
        id.bind_group_layout_ids.iter().try_for_each(|id| f(BindGroupLayout::upcast(id.borrow())))
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // &self.life_guard
    }
}

impl<A: hal::Api> Drop for PipelineLayout<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the pipeline layout is uniquely owned, so it is unused by any CPU
                // resources, and the rest of the program guarantees that it's not used by
                // any GPU resources either (absent panics), so calling destroy_bind_group is
                // safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.device_id.raw.destroy_pipeline_layout(ManuallyDrop::take(&mut self.raw));
            }
        }
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

impl BufferBinding {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, A: hal::Api, F: AllResources<A>, E>(
        &'b self,
        _f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        // FIXME: Perform when we update BufferId!
        // f(crate::resource::Buffer::upcast(buffer_id))
        Ok(())
    }
}

// Note: Duplicated in `wgpu-rs` as `BindingResource`
// They're different enough that it doesn't make sense to share a common type
#[derive(Debug/*, Clone*/)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum BindingResource<'a, A: hal::Api, F: AllResources<A>> {
    Buffer(BufferBinding),
    BufferArray(Cow<'a, [BufferBinding]>),
    Sampler(
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F as Hkt<crate::resource::Sampler<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F as Hkt<crate::resource::Sampler<A>>>::Output: serde::Deserialize<'de>")))]
    <F as Hkt<crate::resource::Sampler<A>>>::Output),
    TextureView(TextureViewId),
    TextureViewArray(Cow<'a, [TextureViewId]>),
}

impl<'a, A: hal::Api, F: AllResources<A>> BindingResource<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(&'b self, mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>) -> Result<(), E> {
        use BindingResource::*;

        match self {
            Buffer(bb) => bb.trace_resources(f),
            BufferArray(bindings_array) => bindings_array.iter().try_for_each(|bb| bb.trace_resources(&mut f)),
            Sampler(id) => f(crate::resource::Sampler::upcast(id)),
            TextureView(_id) => {
                // FIXME: Perform when we update TextureViewId!
                // f(crate::resource::TextureView::upcast(id))
                Ok(())
            },
            TextureViewArray(bindings_array) => bindings_array.iter().try_for_each(|_id| {
                // FIXME: Perform when we update TextureViewId!
                // f(crate::resource::TextureView::upcast(id))
                Ok(())
            }),
        }
    }
}

#[cfg(feature = "trace")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<BindingResource<'a, A, F>> for BindingResource<'b, B, G>
    where
        <F as Hkt<crate::resource::Sampler<A>>>::Output:
            Into<<G as Hkt<crate::resource::Sampler<B>>>::Output>,
{
    fn from(desc: BindingResource<'a, A, F>) -> Self {
        use BindingResource::*;

        match desc {
            Buffer(bb) => Buffer(bb),
            BufferArray(bindings_array) => BufferArray(bindings_array),
            Sampler(id) => Sampler(id.into()),
            TextureView(id) => TextureView(id),
            TextureViewArray(bindings_array) => TextureViewArray(bindings_array),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, &'b BindingResource<'a, A, F>)> for BindingResource<'b, B, G>
    where
        (&'b id::IdCache2, &'b <F as Hkt<crate::resource::Sampler<A>>>::Output):
            TryInto<<G as Hkt<crate::resource::Sampler<B>>>::Output, Error=E>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, &'b BindingResource<'a, A, F>)) -> Result<Self, Self::Error> {
        use BindingResource::*;

        Ok(match desc {
            Buffer(bb) => Buffer(bb.clone()),
            BufferArray(bindings_array) => BufferArray(Cow::Borrowed(&*bindings_array)),
            Sampler(id) => Sampler((cache, id).try_into()?),
            TextureView(id) => TextureView(*id),
            TextureViewArray(bindings_array) => TextureViewArray(Cow::Borrowed(&*bindings_array)),
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum BindError {
    #[error("number of dynamic offsets ({actual}) doesn't match the number of dynamic bindings in the bind group layout ({expected})")]
    MismatchedDynamicOffsetCount { actual: usize, expected: usize },
    #[error(
        "dynamic binding at index {idx}: offset {offset} does not respect `BIND_BUFFER_ALIGNMENT`"
    )]
    UnalignedDynamicBinding { idx: usize, offset: u32 },
    #[error("dynamic binding at index {idx} with offset {offset} would overrun the buffer (limit: {max})")]
    DynamicBindingOutOfBounds { idx: usize, offset: u32, max: u64 },
}

#[derive(Debug)]
pub struct BindGroupDynamicBindingData {
    /// The maximum value the dynamic offset can have before running off the end of the buffer.
    pub(crate) maximum_dynamic_offset: wgt::BufferAddress,
}

#[derive(Debug)]
pub struct BindGroup<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::BindGroup>,
    pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    /// Unfortunately, even though we don't need to access the BindGroup through this ID, we need
    /// to hold onto it to make sure (i) deduplication semantics are respected, and (ii) for memory
    /// safety (the Vulkan spec requires both the pipeline layout and binding group to be pointing
    /// to BGLs that are alive on command buffer submission, even though in theory only one should
    /// be needed).
    pub(crate) layout_id: id::ValidId2<BindGroupLayout<A>>,
    // pub(crate) life_guard: LifeGuard,
    pub(crate) used: TrackerSet<A>,
    pub(crate) used_buffer_ranges: Vec<MemoryInitTrackerAction<BufferId>>,
    pub(crate) dynamic_binding_info: Vec<BindGroupDynamicBindingData>,
}

impl<A: hal::Api> BindGroup<A> {
    pub(crate) fn validate_dynamic_bindings(
        &self,
        offsets: &[wgt::DynamicOffset],
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
            if offset as wgt::BufferAddress % wgt::BIND_BUFFER_ALIGNMENT != 0 {
                return Err(BindError::UnalignedDynamicBinding { idx, offset });
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

/* impl<A: hal::Api> Borrow<()> for BindGroup<A> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
} */

impl<A: hal::Api> Resource for BindGroup<A>
/*where for<'c> id::IdGuardCon<'c>: Hkt<Self, Output=id::IdGuard<'c, A, BindGroup<id::Dummy>>> */{
    const TYPE: &'static str = "BindGroup";

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        id: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        mut f: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: crate::hub::HalApi + 'b,
    {
        f(BindGroupLayout::upcast(id.layout_id.borrow()))?;
        // Does *not* trace `used_buffer_ranges`, because they are already covered by `used`.
        id.used.trace_resources(f)
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // &self.life_guard
    }
}

impl<A: hal::Api> Drop for BindGroup<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the bind group is uniquely owned, so it is unused by any CPU resources,
                // and the rest of the program guarantees that it's not used by any GPU resources
                // either (absent panics), so calling destroy_bind_group is safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.device_id.raw.destroy_bind_group(ManuallyDrop::take(&mut self.raw));
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum GetBindGroupLayoutError {
    #[error("pipeline is invalid")]
    InvalidPipeline,
    #[error("invalid group index {0}")]
    InvalidGroupIndex(u32),
}
