use crate::{Epoch, Index};
use std::{
    cmp::Ordering,
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
};
use wgt::{Backend, WasmNotSendSync};

type IdType = u64;
type ZippedIndex = Index;
type NonZeroId = std::num::NonZeroU64;

const INDEX_BITS: usize = ZippedIndex::BITS as usize;
const EPOCH_BITS: usize = INDEX_BITS - BACKEND_BITS;
const BACKEND_BITS: usize = 3;
const BACKEND_SHIFT: usize = INDEX_BITS * 2 - BACKEND_BITS;
pub const EPOCH_MASK: u32 = (1 << (EPOCH_BITS)) - 1;

/// The raw underlying representation of an identifier.
#[repr(transparent)]
#[cfg_attr(
    any(feature = "serde", feature = "trace"),
    derive(serde::Serialize),
    serde(into = "SerialId")
)]
#[cfg_attr(
    any(feature = "serde", feature = "replay"),
    derive(serde::Deserialize),
    serde(from = "SerialId")
)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawId(NonZeroId);

impl RawId {
    #[doc(hidden)]
    #[inline]
    pub fn from_non_zero(non_zero: NonZeroId) -> Self {
        Self(non_zero)
    }

    #[doc(hidden)]
    #[inline]
    pub fn into_non_zero(self) -> NonZeroId {
        self.0
    }

    /// Zip together an identifier and return its raw underlying representation.
    pub fn zip(index: Index, epoch: Epoch, backend: Backend) -> RawId {
        assert_eq!(0, epoch >> EPOCH_BITS);
        assert_eq!(0, (index as IdType) >> INDEX_BITS);
        let v = index as IdType
            | ((epoch as IdType) << INDEX_BITS)
            | ((backend as IdType) << BACKEND_SHIFT);
        Self(NonZeroId::new(v).unwrap())
    }

    /// Unzip a raw identifier into its components.
    #[allow(trivial_numeric_casts)]
    pub fn unzip(self) -> (Index, Epoch, Backend) {
        (
            (self.0.get() as ZippedIndex) as Index,
            (((self.0.get() >> INDEX_BITS) as ZippedIndex) & (EPOCH_MASK as ZippedIndex)) as Index,
            self.backend(),
        )
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

/// An identifier for a wgpu object.
///
/// An `Id<T>` value identifies a value stored in a [`Global`]'s [`Hub`].
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
/// [`Texture<A>`]: crate::resource::Texture
/// [`Registry`]: crate::hub::Registry
/// [`Empty`]: hal::api::Empty
#[repr(transparent)]
#[cfg_attr(any(feature = "serde", feature = "trace"), derive(serde::Serialize))]
#[cfg_attr(any(feature = "serde", feature = "replay"), derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serde", feature = "trace", feature = "replay"),
    serde(transparent)
)]
pub struct Id<T: Marker>(RawId, PhantomData<T>);

// This type represents Id in a more readable (and editable) way.
#[allow(dead_code)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum SerialId {
    // The only variant forces RON to not ignore "Id"
    Id(Index, Epoch, Backend),
}

impl From<RawId> for SerialId {
    fn from(id: RawId) -> Self {
        let (index, epoch, backend) = id.unzip();
        Self::Id(index, epoch, backend)
    }
}

impl From<SerialId> for RawId {
    fn from(id: SerialId) -> Self {
        match id {
            SerialId::Id(index, epoch, backend) => RawId::zip(index, epoch, backend),
        }
    }
}

impl<T> Id<T>
where
    T: Marker,
{
    /// # Safety
    ///
    /// The raw id must be valid for the type.
    pub unsafe fn from_raw(raw: RawId) -> Self {
        Self(raw, PhantomData)
    }

    /// Coerce the identifiers into its raw underlying representation.
    pub fn into_raw(self) -> RawId {
        self.0
    }

    #[allow(dead_code)]
    pub(crate) fn dummy(index: u32) -> Self {
        Id::zip(index, 1, Backend::Empty)
    }

    #[allow(dead_code)]
    pub(crate) fn is_valid(&self) -> bool {
        self.backend() != Backend::Empty
    }

    /// Get the backend this identifier corresponds to.
    #[inline]
    pub fn backend(self) -> Backend {
        self.0.backend()
    }

    #[inline]
    pub fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self {
        Id(RawId::zip(index, epoch, backend), PhantomData)
    }

    #[inline]
    pub fn unzip(self) -> (Index, Epoch, Backend) {
        self.0.unzip()
    }
}

impl<T> Copy for Id<T> where T: Marker {}

impl<T> Clone for Id<T>
where
    T: Marker,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Debug for Id<T>
where
    T: Marker,
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
    T: Marker,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T>
where
    T: Marker,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> where T: Marker {}

impl<T> PartialOrd for Id<T>
where
    T: Marker,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T>
where
    T: Marker,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Marker trait used to determine which types uniquely identify a resource.
///
/// For example, `Device<A>` will have the same type of identifier as
/// `Device<B>` because `Device<T>` for any `T` defines the same maker type.
pub trait Marker: 'static + WasmNotSendSync {}

// This allows `()` to be used as a marker type for tests.
//
// We don't want these in production code, since they essentially remove type
// safety, like how identifiers across different types can be compared.
#[cfg(test)]
impl Marker for () {}

/// Define identifiers for each resource.
macro_rules! ids {
    ($(
        $(#[$($meta:meta)*])*
        pub type $name:ident $marker:ident;
    )*) => {
        /// Marker types for each resource.
        pub mod markers {
            $(
                #[derive(Debug)]
                pub enum $marker {}
                impl super::Marker for $marker {}
            )*
        }

        $(
            $(#[$($meta)*])*
            pub type $name = Id<self::markers::$marker>;
        )*
    }
}

ids! {
    pub type AdapterId Adapter;
    pub type SurfaceId Surface;
    pub type DeviceId Device;
    pub type QueueId Queue;
    pub type BufferId Buffer;
    pub type StagingBufferId StagingBuffer;
    pub type TextureViewId TextureView;
    pub type TextureId Texture;
    pub type SamplerId Sampler;
    pub type BindGroupLayoutId BindGroupLayout;
    pub type PipelineLayoutId PipelineLayout;
    pub type BindGroupId BindGroup;
    pub type ShaderModuleId ShaderModule;
    pub type RenderPipelineId RenderPipeline;
    pub type ComputePipelineId ComputePipeline;
    pub type PipelineCacheId PipelineCache;
    pub type CommandEncoderId CommandEncoder;
    pub type CommandBufferId CommandBuffer;
    pub type RenderPassEncoderId RenderPassEncoder;
    pub type ComputePassEncoderId ComputePassEncoder;
    pub type RenderBundleEncoderId RenderBundleEncoder;
    pub type RenderBundleId RenderBundle;
    pub type QuerySetId QuerySet;
}

// The CommandBuffer type serves both as encoder and
// buffer, which is why the 2 functions below exist.

impl CommandEncoderId {
    pub fn into_command_buffer_id(self) -> CommandBufferId {
        Id(self.0, PhantomData)
    }
}

impl CommandBufferId {
    pub fn into_command_encoder_id(self) -> CommandEncoderId {
        Id(self.0, PhantomData)
    }
}

#[test]
fn test_id_backend() {
    for &b in &[
        Backend::Empty,
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Gl,
    ] {
        let id = Id::<()>::zip(1, 0, b);
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
                let id = Id::<()>::zip(i, e, b);
                let (index, epoch, backend) = id.unzip();
                assert_eq!(index, i);
                assert_eq!(epoch, e);
                assert_eq!(backend, b);
            }
        }
    }
}
