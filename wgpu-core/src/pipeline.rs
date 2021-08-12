#[cfg(feature = "replay")] use core::convert::{TryFrom, TryInto};
use crate::{
    binding_model::{CreateBindGroupLayoutError, CreatePipelineLayoutError, PipelineLayout},
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures, RenderPassContext},
    hub::Resource,
    id::{self, AnyBackend, AllResources, Hkt},
    validation, Label, LifeGuard,
};
#[cfg(feature = "trace")]
use crate::{
    command::FromCommand,
    id::BorrowHkt,
};
use hal::{Device as _};
use std::{borrow::Cow, mem::ManuallyDrop};
use thiserror::Error;

pub enum ShaderModuleSource<'a> {
    Wgsl(Cow<'a, str>),
    Naga(naga::Module),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ShaderModuleDescriptor<'a> {
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub label: Label<'a>,
}

impl<'a> ShaderModuleDescriptor<'a> {
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

#[derive(Debug)]
pub struct ShaderModule<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::ShaderModule>,
    pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    pub(crate) interface: Option<validation::Interface>,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl<A: hal::Api> Resource for ShaderModule<A> {
    const TYPE: &'static str = "ShaderModule";

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
        unreachable!()
    }

    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

impl<A: hal::Api> Drop for ShaderModule<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the shader module is uniquely owned, so it is unused by any CPU
                // resources, and shader modules may be dropped even while pipelines derived from
                // them are active on the GPU, so calling destroy_shader_module is safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.device_id.raw.destroy_shader_module(ManuallyDrop::take(&mut self.raw));
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
pub struct NagaParseError {
    pub shader_source: String,
    pub error: naga::front::wgsl::ParseError,
}
impl std::fmt::Display for NagaParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\nShader error:\n{}",
            self.error.emit_to_string(&self.shader_source)
        )
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreateShaderModuleError {
    #[error("Failed to parse a shader")]
    Parsing(#[from] NagaParseError),
    #[error("Failed to generate the backend-specific code")]
    Generation,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    Validation(#[from] naga::valid::ValidationError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

/// Describes a programmable pipeline stage.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ProgrammableStageDescriptor<'a, A: hal::Api, F: AllResources<A>> {
    /// The compiled shader module for this stage.

    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F as Hkt<ShaderModule<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F as Hkt<ShaderModule<A>>>::Output: serde::Deserialize<'de>")))]
    pub module: /*ShaderModule<A>*/<F as Hkt<ShaderModule<A>>>::Output,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub entry_point: Cow<'a, str>,
}

impl<'a, A: hal::Api, F: AllResources<A>> ProgrammableStageDescriptor<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        f(ShaderModule::upcast(&self.module))
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b ProgrammableStageDescriptor<'a, A, F>> for ProgrammableStageDescriptor<'b, B, G>
    where
        &'b <F as Hkt<ShaderModule<A>>>::Output:
            Into<<G as Hkt<ShaderModule<B>>>::Output>,
{
    fn from(desc: &'b ProgrammableStageDescriptor<'a, A, F>) -> Self {
        Self {
            module: (&desc.module).into(),
            entry_point: Cow::Borrowed(&*desc.entry_point),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, ProgrammableStageDescriptor<'a, A, F>)> for ProgrammableStageDescriptor<'b, B, G>
    where
        (&'b id::IdCache2, <F as Hkt<ShaderModule<A>>>::Output):
            TryInto<<G as Hkt<ShaderModule<B>>>::Output, Error=E>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, ProgrammableStageDescriptor<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            module: (cache, desc.module).try_into()?,
            entry_point: desc.entry_point,
        })
    }
}

/// Number of implicit bind groups derived at pipeline creation.
pub type ImplicitBindGroupCount = u8;

#[derive(Clone, Debug, Error)]
pub enum ImplicitLayoutError {
    /* #[error("missing IDs for deriving {0} bind groups")]
    MissingIds(ImplicitBindGroupCount), */
    #[error("unable to reflect the shader {0:?} interface")]
    ReflectionError(wgt::ShaderStages),
    #[error(transparent)]
    BindGroup(#[from] CreateBindGroupLayoutError),
    #[error(transparent)]
    Pipeline(#[from] CreatePipelineLayoutError),
}

/// Describes a compute pipeline.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ComputePipelineDescriptor<'a, A: hal::Api, F: AllResources<A>> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F::Owned as Hkt<PipelineLayout<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F::Owned as Hkt<PipelineLayout<A>>>::Output: serde::Deserialize<'de>")))]
    pub layout: Option</*PipelineLayoutId*/<F::Owned as Hkt<PipelineLayout<A>>>::Output>,
    /// The compiled compute stage and its entry point.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Deserialize<'de>")))]
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub stage: ProgrammableStageDescriptor<'a, A, F>,
}

impl<'a, A: hal::Api, F: AllResources<A>> ComputePipelineDescriptor<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where
            id::Cached<A, &'b F>: 'b,
            <&'b F::Owned as Hkt<PipelineLayout<A>>>::Output:
                Into<&'b <F as Hkt<PipelineLayout<A>>>::Output>,
    {
        self.layout.as_ref().map(|id| f(PipelineLayout::upcast(/*<&F>::borrow(id))*/id.into()))).transpose()?;
        self.stage.trace_resources(f)
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b ComputePipelineDescriptor<'a, A, F>> for ComputePipelineDescriptor<'b, B, G>
    where
        ProgrammableStageDescriptor<'a, B, G>:
            FromCommand<&'b ProgrammableStageDescriptor<'a, A, F>>,
        A: crate::hub::HalApi,
        G::Owned: BorrowHkt<A, PipelineLayout<B>, PipelineLayout<hal::api::Empty>, F::Owned>,
{
    fn from(desc: &'b ComputePipelineDescriptor<'a, A, F>) -> Self {
        Self {
            label: desc.label.as_deref().map(Cow::Borrowed),
            layout: desc.layout.as_ref().map(G::Owned::borrow),
            stage: FromCommand::from(&desc.stage),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'a id::IdCache2, ComputePipelineDescriptor<'a, A, F>)> for ComputePipelineDescriptor<'a, B, G>
    where
        (&'a id::IdCache2, ProgrammableStageDescriptor<'a, A, F>):
            TryInto<ProgrammableStageDescriptor<'a, B, G>, Error=E>,
        (&'a id::IdCache2, <F::Owned as Hkt<PipelineLayout<A>>>::Output):
            TryInto<<G as Hkt<PipelineLayout<B>>>::Output, Error=E>,
        <G as Hkt<PipelineLayout<B>>>::Output:
            Into<<G::Owned as Hkt<PipelineLayout<B>>>::Output>,
{
    type Error = E;

    fn try_from((cache, desc): (&'a id::IdCache2, ComputePipelineDescriptor<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            label: desc.label,
            layout: desc.layout
                .map(|layout| TryInto::<<G as Hkt<PipelineLayout<B>>>::Output>::try_into((cache, layout)))
                .transpose()?
                .map(Into::into),
            stage: (cache, desc.stage).try_into()?,
        })
    }
}

pub type ComputePipelineDescriptorIn<'a, A> = ComputePipelineDescriptor<'a, A, id::IdGuardCon<'a>>;

#[derive(Clone, Debug, Error)]
pub enum CreateComputePipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("pipeline layout is invalid")]
    InvalidLayout,
    #[error("unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error("error matching shader requirements against the pipeline")]
    Stage(#[from] validation::StageError),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

#[derive(Debug)]
pub struct ComputePipeline<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::ComputePipeline>,
    pub(crate) layout_id: /*Stored<PipelineLayoutId>*/id::ValidId2<PipelineLayout<A>>,
    // /// NOTE: Already available on the layout, so it's redundant here; if device is removed from
    // /// internal resources, it won't be, but in that case it won't be in either structure.
    // pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    // pub(crate) life_guard: LifeGuard,
}

impl<A: hal::Api> Resource for ComputePipeline<A> {
    const TYPE: &'static str = "ComputePipeline";

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        id: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        mut f: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: crate::hub::HalApi + 'b,
    {
        f(PipelineLayout::upcast(id.layout_id.borrow()))
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // &self.life_guard
    }
}

impl<A: hal::Api> Drop for ComputePipeline<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the compute pipeline is uniquely owned, so it is unused by any CPU
                // resources, and the rest of the program guarantees that it's not used by any GPU
                // resources either (absent panics), so calling destroy_sampler is safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.layout_id.device_id.raw.destroy_compute_pipeline(ManuallyDrop::take(&mut self.raw));
            }
        }
    }
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: wgt::BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: wgt::VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub attributes: Cow<'a, [wgt::VertexAttribute]>,
}

/// Describes the vertex process in a render pipeline.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct VertexState<'a, A: hal::Api, F: AllResources<A>> {
    /// The compiled vertex stage and its entry point.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Deserialize<'de>")))]
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub stage: ProgrammableStageDescriptor<'a, A, F>,
    /// The format of any vertex buffers used with this pipeline.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub buffers: Cow<'a, [VertexBufferLayout<'a>]>,
}

impl<'a, A: hal::Api, F: AllResources<A>> VertexState<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        self.stage.trace_resources(f)
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b VertexState<'a, A, F>> for VertexState<'b, B, G>
    where
        ProgrammableStageDescriptor<'b, B, G>:
            FromCommand<&'b ProgrammableStageDescriptor<'a, A, F>>,
{
    fn from(desc: &'b VertexState<'a, A, F>) -> Self {
        Self {
            stage: FromCommand::from(&desc.stage),
            buffers: Cow::Borrowed(&*desc.buffers),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, VertexState<'a, A, F>)> for VertexState<'b, B, G>
    where
        (&'b id::IdCache2, ProgrammableStageDescriptor<'a, A, F>):
            TryInto<ProgrammableStageDescriptor<'b, B, G>, Error=E>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, VertexState<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            stage: (cache, desc.stage).try_into()?,
            // NB: Weird variance issue, I think on reflection it should actually be safe to just
            // use desc.buffers (and consequently not need to differentiate between `'a` and `'b`
            // here or in the other `TryFrom` implementations).  But rustc won't let us, which I
            // think may be a bug.
            buffers: /*Cow::Borrowed(desc.buffers.borrow())*/
                match desc.buffers {
                    Cow::Borrowed(buffers) => Cow::Borrowed(buffers),
                    Cow::Owned(buffers) => Cow::Owned(buffers),
                }
        })
    }
}

/// Describes fragment processing in a render pipeline.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct FragmentState<'a, A: hal::Api, F: AllResources<A>> {
    /// The compiled fragment stage and its entry point.
    #[cfg_attr(feature = "trace",
      serde(bound(serialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(feature = "replay",
      serde(bound(deserialize = "ProgrammableStageDescriptor<'a, A, F>: serde::Deserialize<'de>")))]
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub stage: ProgrammableStageDescriptor<'a, A, F>,
    /// The effect of draw calls on the color aspect of the output target.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub targets: Cow<'a, [wgt::ColorTargetState]>,
}

impl<'a, A: hal::Api, F: AllResources<A>> FragmentState<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        self.stage.trace_resources(f)
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b FragmentState<'a, A, F>> for FragmentState<'b, B, G>
    where
        ProgrammableStageDescriptor<'b, B, G>:
            FromCommand<&'b ProgrammableStageDescriptor<'a, A, F>>,
{
    fn from(desc: &'b FragmentState<'a, A, F>) -> Self {
        Self {
            stage: FromCommand::from(&desc.stage),
            targets: Cow::Borrowed(&*desc.targets),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, FragmentState<'a, A, F>)> for FragmentState<'b, B, G>
    where
        (&'b id::IdCache2, ProgrammableStageDescriptor<'a, A, F>):
            TryInto<ProgrammableStageDescriptor<'b, B, G>, Error=E>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, FragmentState<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            stage: (cache, desc.stage).try_into()?,
            targets: desc.targets,
        })
    }
}

/// Describes a render (graphics) pipeline.
#[derive(/*Clone, */Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct RenderPipelineDescriptor<'a, A: hal::Api, F: AllResources<A>> {
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "<F::Owned as Hkt<PipelineLayout<A>>>::Output: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "<F::Owned as Hkt<PipelineLayout<A>>>::Output: serde::Deserialize<'de>")))]
    pub layout: Option</*PipelineLayoutId*/<F::Owned as Hkt<PipelineLayout<A>>>::Output>,
    /// The vertex processing state for this pipeline.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "VertexState<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "VertexState<'a, A, F>: serde::Deserialize<'de>")))]
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub vertex: VertexState<'a, A, F>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
    pub multisample: wgt::MultisampleState,
    /// The fragment processing state for this pipeline.
    #[cfg_attr(any(feature = "trace"),
      serde(bound(serialize = "FragmentState<'a, A, F>: serde::Serialize")))]
    #[cfg_attr(any(feature = "replay"),
      serde(bound(deserialize = "FragmentState<'a, A, F>: serde::Deserialize<'de>")))]
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(borrow))]
    pub fragment: Option<FragmentState<'a, A, F>>,
}

impl<'a, A: hal::Api, F: AllResources<A>> RenderPipelineDescriptor<'a, A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where
            id::Cached<A, &'b F>: 'b,
            <&'b F::Owned as Hkt<PipelineLayout<A>>>::Output:
                Into<&'b <F as Hkt<PipelineLayout<A>>>::Output>,
    {
        self.layout.as_ref().map(|id| f(PipelineLayout::upcast(/*<&F>::borrow(id))*/id.into()))).transpose()?;
        self.fragment.as_ref().map(|frag| frag.trace_resources(&mut f)).transpose()?;
        self.vertex.trace_resources(f)
    }
}

#[cfg(feature = "trace")]
impl<'a, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    FromCommand<&'b RenderPipelineDescriptor<'a, A, F>> for RenderPipelineDescriptor<'b, B, G>
    where
        VertexState<'b, B, G>:
            FromCommand<&'b VertexState<'a, A, F>>,
        FragmentState<'b, B, G>:
            FromCommand<&'b FragmentState<'a, A, F>>,
        A: crate::hub::HalApi,
        G::Owned: BorrowHkt<A, PipelineLayout<B>, PipelineLayout<hal::api::Empty>, F::Owned>,
{
    fn from(desc: &'b RenderPipelineDescriptor<'a, A, F>) -> Self {
        Self {
            label: desc.label.as_deref().map(Cow::Borrowed),
            layout: desc.layout.as_ref().map(G::Owned::borrow),
            vertex: FromCommand::from(&desc.vertex),
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(FromCommand::from),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a: 'b, 'b, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    TryFrom<(&'b id::IdCache2, RenderPipelineDescriptor<'a, A, F>)> for RenderPipelineDescriptor<'b, B, G>
    where
        (&'b id::IdCache2, VertexState<'a, A, F>):
            TryInto<VertexState<'b, B, G>, Error=E>,
        (&'b id::IdCache2, FragmentState<'a, A, F>):
            TryInto<FragmentState<'b, B, G>, Error=E>,
        (&'b id::IdCache2, <F::Owned as Hkt<PipelineLayout<A>>>::Output):
            TryInto<<G as Hkt<PipelineLayout<B>>>::Output, Error=E>,
        <G as Hkt<PipelineLayout<B>>>::Output:
            Into<<G::Owned as Hkt<PipelineLayout<B>>>::Output>,
{
    type Error = E;

    fn try_from((cache, desc): (&'b id::IdCache2, RenderPipelineDescriptor<'a, A, F>)) -> Result<Self, Self::Error> {
        Ok(Self {
            label: desc.label,
            layout: desc.layout
                .map(|layout| TryInto::<<G as Hkt<PipelineLayout<B>>>::Output>::try_into((cache, layout)))
                .transpose()?
                .map(Into::into),
            vertex: (cache, desc.vertex).try_into()?,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil,
            multisample: desc.multisample,
            fragment: desc.fragment.map(|frag| (cache, frag).try_into()).transpose()?,
        })
    }
}

pub type RenderPipelineDescriptorIn<'a, A> = RenderPipelineDescriptor<'a, A, id::IdGuardCon<'a>>;

#[derive(Clone, Debug, Error)]
pub enum ColorStateError {
    #[error("output is missing")]
    Missing,
    #[error("format {0:?} is not renderable")]
    FormatNotRenderable(wgt::TextureFormat),
    #[error("format {0:?} is not blendable")]
    FormatNotBlendable(wgt::TextureFormat),
    #[error("format {0:?} does not have a color aspect")]
    FormatNotColor(wgt::TextureFormat),
    #[error("output format {pipeline} is incompatible with the shader {shader}")]
    IncompatibleFormat {
        pipeline: validation::NumericType,
        shader: validation::NumericType,
    },
    #[error("blend factors for {0:?} must be `One`")]
    InvalidMinMaxBlendFactors(wgt::BlendComponent),
}

#[derive(Clone, Debug, Error)]
pub enum DepthStencilStateError {
    #[error("format {0:?} is not renderable")]
    FormatNotRenderable(wgt::TextureFormat),
    #[error("format {0:?} does not have a depth aspect, but depth test/write is enabled")]
    FormatNotDepth(wgt::TextureFormat),
    #[error("format {0:?} does not have a stencil aspect, but stencil test/write is enabled")]
    FormatNotStencil(wgt::TextureFormat),
}

#[derive(Clone, Debug, Error)]
pub enum CreateRenderPipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("pipeline layout is invalid")]
    InvalidLayout,
    #[error("unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error("color state [{0}] is invalid")]
    ColorState(u8, #[source] ColorStateError),
    #[error("depth/stencil state is invalid")]
    DepthStencilState(#[from] DepthStencilStateError),
    #[error("invalid sample count {0}")]
    InvalidSampleCount(u32),
    #[error("the number of vertex buffers {given} exceeds the limit {limit}")]
    TooManyVertexBuffers { given: u32, limit: u32 },
    #[error("the total number of vertex attributes {given} exceeds the limit {limit}")]
    TooManyVertexAttributes { given: u32, limit: u32 },
    #[error("vertex buffer {index} stride {given} exceeds the limit {limit}")]
    VertexStrideTooLarge { index: u32, given: u32, limit: u32 },
    #[error("vertex buffer {index} stride {stride} does not respect `VERTEX_STRIDE_ALIGNMENT`")]
    UnalignedVertexStride {
        index: u32,
        stride: wgt::BufferAddress,
    },
    #[error("vertex attribute at location {location} has invalid offset {offset}")]
    InvalidVertexAttributeOffset {
        location: wgt::ShaderLocation,
        offset: wgt::BufferAddress,
    },
    #[error("strip index format was not set to None but to {strip_index_format:?} while using the non-strip topology {topology:?}")]
    StripIndexFormatForNonStripTopology {
        strip_index_format: Option<wgt::IndexFormat>,
        topology: wgt::PrimitiveTopology,
    },
    #[error("Conservative Rasterization is only supported for wgt::PolygonMode::Fill")]
    ConservativeRasterizationNonFillPolygonMode,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("error matching {stage:?} shader requirements against the pipeline")]
    Stage {
        stage: wgt::ShaderStages,
        #[source]
        error: validation::StageError,
    },
    #[error("Internal error in {stage:?} shader: {error}")]
    Internal {
        stage: wgt::ShaderStages,
        error: String,
    },
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PipelineFlags: u32 {
        const BLEND_CONSTANT = 1 << 0;
        const STENCIL_REFERENCE = 1 << 1;
        const WRITES_DEPTH_STENCIL = 1 << 2;
    }
}

#[derive(Debug)]
pub struct RenderPipeline<A: hal::Api> {
    pub(crate) raw: ManuallyDrop<A::RenderPipeline>,
    pub(crate) layout_id: /*Stored<PipelineLayoutId>*/id::ValidId2<PipelineLayout<A>>,
    // /// NOTE: Already available on the layout, so it's redundant here; if device is removed from
    // /// internal resources, it won't be, but in that case it won't be in either structure.
    // pub(crate) device_id: /*Stored<DeviceId>*/id::ValidId2<Device<A>>,
    pub(crate) pass_context: RenderPassContext,
    pub(crate) flags: PipelineFlags,
    pub(crate) strip_index_format: Option<wgt::IndexFormat>,
    pub(crate) vertex_strides: Vec<(wgt::BufferAddress, wgt::VertexStepMode)>,
    // pub(crate) life_guard: LifeGuard,
}

impl<A: hal::Api> Resource for RenderPipeline<A> {
    const TYPE: &'static str = "RenderPipeline";

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        id: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        mut f: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: crate::hub::HalApi + 'b,
    {
        f(PipelineLayout::upcast(id.layout_id.borrow()))
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // &self.life_guard
    }
}

impl<A: hal::Api> Drop for RenderPipeline<A> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                // Safety: the render pipeline is uniquely owned, so it is unused by any CPU
                // resources, and the rest of the program guarantees that it's not used by any GPU
                // resources either (absent panics), so calling destroy_sampler is safe.
                //
                // We never use self.raw again after calling ManuallyDrop::take, so calling that
                // is also safe.
                self.layout_id.device_id.raw.destroy_render_pipeline(ManuallyDrop::take(&mut self.raw));
            }
        }
    }
}
