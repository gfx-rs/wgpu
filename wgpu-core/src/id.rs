use crate::{Epoch, Index};
use std::{
    any::Any,
    cmp::Ordering,
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
};
use wgt::{Backend, WasmNotSendSync};

type IdType = u64;
type NonZeroId = std::num::NonZeroU64;
type ZippedIndex = Index;

const INDEX_BITS: usize = std::mem::size_of::<ZippedIndex>() * 8;
const EPOCH_BITS: usize = INDEX_BITS - BACKEND_BITS;
const BACKEND_BITS: usize = 3;
const BACKEND_SHIFT: usize = INDEX_BITS * 2 - BACKEND_BITS;
pub const EPOCH_MASK: u32 = (1 << (EPOCH_BITS)) - 1;
type Dummy = hal::api::Empty;

/// An identifier for a wgpu object.
///
/// An `Id<T>` value identifies a value stored in a [`Global`]'s [`Hub`]'s [`Storage`].
/// `Storage` implements [`Index`] and [`IndexMut`], accepting `Id` values as indices.
///
/// ## Note on `Id` typing
///
/// You might assume that an `Id<T>` can only be used to retrieve a resource of
/// type `T`, but that is not quite the case. The id types in `wgpu-core`'s
/// public API ([`TextureId`], for example) can refer to resources belonging to
/// any backend, but the corresponding resource types ([`Texture<A>`], for
/// example) are always parameterized by a specific backend `A`.
///
/// So the `T` in `Id<T>` is usually a resource type like `Texture<Empty>`,
/// where [`Empty`] is the `wgpu_hal` dummy back end. These empty types are
/// never actually used, beyond just making sure you access each `Storage` with
/// the right kind of identifier. The members of [`Hub<A>`] pair up each
/// `X<Empty>` type with the resource type `X<A>`, for some specific backend
/// `A`.
///
/// [`Global`]: crate::global::Global
/// [`Hub`]: crate::hub::Hub
/// [`Hub<A>`]: crate::hub::Hub
/// [`Storage`]: crate::storage::Storage
/// [`Texture<A>`]: crate::resource::Texture
/// [`Index`]: std::ops::Index
/// [`IndexMut`]: std::ops::IndexMut
/// [`Registry`]: crate::hub::Registry
/// [`Empty`]: hal::api::Empty
#[repr(transparent)]
#[cfg_attr(feature = "trace", derive(serde::Serialize), serde(into = "SerialId"))]
#[cfg_attr(
    feature = "replay",
    derive(serde::Deserialize),
    serde(from = "SerialId")
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "trace")),
    derive(serde::Serialize)
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "replay")),
    derive(serde::Deserialize)
)]
pub struct Id<T: 'static + WasmNotSendSync>(NonZeroId, PhantomData<T>);

// This type represents Id in a more readable (and editable) way.
#[allow(dead_code)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
enum SerialId {
    // The only variant forces RON to not ignore "Id"
    Id(Index, Epoch, Backend),
}
#[cfg(feature = "trace")]
impl<T> From<Id<T>> for SerialId
where
    T: 'static + WasmNotSendSync,
{
    fn from(id: Id<T>) -> Self {
        let (index, epoch, backend) = id.unzip();
        Self::Id(index, epoch, backend)
    }
}
#[cfg(feature = "replay")]
impl<T> From<SerialId> for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn from(id: SerialId) -> Self {
        match id {
            SerialId::Id(index, epoch, backend) => TypedId::zip(index, epoch, backend),
        }
    }
}

impl<T> Id<T>
where
    T: 'static + WasmNotSendSync,
{
    /// # Safety
    ///
    /// The raw id must be valid for the type.
    pub unsafe fn from_raw(raw: NonZeroId) -> Self {
        Self(raw, PhantomData)
    }

    #[allow(dead_code)]
    pub(crate) fn dummy(index: u32) -> Self {
        Id::zip(index, 1, Backend::Empty)
    }

    #[allow(dead_code)]
    pub(crate) fn is_valid(&self) -> bool {
        self.backend() != Backend::Empty
    }

    pub fn backend(self) -> Backend {
        match self.0.get() >> (BACKEND_SHIFT) as u8 {
            0 => Backend::Empty,
            1 => Backend::Vulkan,
            2 => Backend::Metal,
            3 => Backend::Dx12,
            4 => Backend::Gl,
            _ => unreachable!(),
        }
    }
}

impl<T> Copy for Id<T> where T: 'static + WasmNotSendSync {}

impl<T> Clone for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Debug for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let (index, epoch, backend) = self.unzip();
        let backend = match backend {
            Backend::Empty => "_",
            Backend::Vulkan => "vk",
            Backend::Metal => "mtl",
            Backend::Dx12 => "d3d12",
            Backend::Gl => "gl",
            Backend::BrowserWebGpu => "webgpu",
        };
        write!(formatter, "Id({index},{epoch},{backend})")?;
        Ok(())
    }
}

impl<T> Hash for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> where T: 'static + WasmNotSendSync {}

impl<T> PartialOrd for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Trait carrying methods for direct `Id` access.
///
/// Most `wgpu-core` clients should not use this trait. Unusual clients that
/// need to construct `Id` values directly, or access their components, like the
/// WGPU recording player, may use this trait to do so.
pub trait TypedId: Copy + Debug + Any + 'static + WasmNotSendSync + Eq + Hash {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self;
    fn unzip(self) -> (Index, Epoch, Backend);
    fn into_raw(self) -> NonZeroId;
}

#[allow(trivial_numeric_casts)]
impl<T> TypedId for Id<T>
where
    T: 'static + WasmNotSendSync,
{
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self {
        assert_eq!(0, epoch >> EPOCH_BITS);
        assert_eq!(0, (index as IdType) >> INDEX_BITS);
        let v = index as IdType
            | ((epoch as IdType) << INDEX_BITS)
            | ((backend as IdType) << BACKEND_SHIFT);
        Id(NonZeroId::new(v).unwrap(), PhantomData)
    }

    fn unzip(self) -> (Index, Epoch, Backend) {
        (
            (self.0.get() as ZippedIndex) as Index,
            (((self.0.get() >> INDEX_BITS) as ZippedIndex) & (EPOCH_MASK as ZippedIndex)) as Index,
            self.backend(),
        )
    }

    fn into_raw(self) -> NonZeroId {
        self.0
    }
}

pub type AdapterId = Id<crate::instance::Adapter<Dummy>>;
pub type SurfaceId = Id<crate::instance::Surface>;
// Device
pub type DeviceId = Id<crate::device::Device<Dummy>>;
pub type QueueId = DeviceId;
// Resource
pub type BufferId = Id<crate::resource::Buffer<Dummy>>;
pub type StagingBufferId = Id<crate::resource::StagingBuffer<Dummy>>;
pub type TextureViewId = Id<crate::resource::TextureView<Dummy>>;
pub type TextureId = Id<crate::resource::Texture<Dummy>>;
pub type SamplerId = Id<crate::resource::Sampler<Dummy>>;
// Binding model
pub type BindGroupLayoutId = Id<crate::binding_model::BindGroupLayout<Dummy>>;
pub type PipelineLayoutId = Id<crate::binding_model::PipelineLayout<Dummy>>;
pub type BindGroupId = Id<crate::binding_model::BindGroup<Dummy>>;
// Pipeline
pub type ShaderModuleId = Id<crate::pipeline::ShaderModule<Dummy>>;
pub type RenderPipelineId = Id<crate::pipeline::RenderPipeline<Dummy>>;
pub type ComputePipelineId = Id<crate::pipeline::ComputePipeline<Dummy>>;
// Command
pub type CommandEncoderId = CommandBufferId;
pub type CommandBufferId = Id<crate::command::CommandBuffer<Dummy>>;
pub type RenderPassEncoderId = *mut crate::command::RenderPass;
pub type ComputePassEncoderId = *mut crate::command::ComputePass;
pub type RenderBundleEncoderId = *mut crate::command::RenderBundleEncoder;
pub type RenderBundleId = Id<crate::command::RenderBundle<Dummy>>;
pub type QuerySetId = Id<crate::resource::QuerySet<Dummy>>;

#[test]
fn test_id_backend() {
    for &b in &[
        Backend::Empty,
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Gl,
    ] {
        let id: Id<()> = Id::zip(1, 0, b);
        let (_id, _epoch, backend) = id.unzip();
        assert_eq!(id.backend(), b);
        assert_eq!(backend, b);
    }
}

#[test]
fn test_id() {
    let last_index = ((1u64 << INDEX_BITS) - 1) as Index;
    let indexes = [1, last_index / 2 - 1, last_index / 2 + 1, last_index];
    let epochs = [1, EPOCH_MASK / 2 - 1, EPOCH_MASK / 2 + 1, EPOCH_MASK];
    let backends = [
        Backend::Empty,
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Gl,
    ];
    for &i in &indexes {
        for &e in &epochs {
            for &b in &backends {
                let id: Id<()> = Id::zip(i, e, b);
                let (index, epoch, backend) = id.unzip();
                assert_eq!(index, i);
                assert_eq!(epoch, e);
                assert_eq!(backend, b);
            }
        }
    }
}
