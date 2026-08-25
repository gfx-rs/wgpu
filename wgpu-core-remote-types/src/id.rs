use crate::{Epoch, Index};
use core::{
    cmp::Ordering,
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
    num::NonZeroU64,
};
pub use wgpu_types::markers::*;

const _: () = {
    if size_of::<Index>() != 4 {
        panic!()
    }
};
const _: () = {
    if size_of::<Epoch>() != 4 {
        panic!()
    }
};
const _: () = {
    if size_of::<RawId>() != 8 {
        panic!()
    }
};

/// The raw underlying representation of an identifier.
#[repr(transparent)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(into = "SerialId", try_from = "SerialId")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RawId(NonZeroU64);

impl RawId {
    /// Zip together an identifier and return its raw underlying representation.
    ///
    /// # Panics
    ///
    /// If both ID components are zero.
    pub fn zip(index: Index, epoch: Epoch) -> RawId {
        let v = (index as u64) | ((epoch as u64) << 32);
        Self(NonZeroU64::new(v).expect("IDs may not be zero"))
    }

    /// Unzip a raw identifier into its components.
    pub fn unzip(self) -> (Index, Epoch) {
        (self.0.get() as Index, (self.0.get() >> 32) as Epoch)
    }
}

/// An identifier for a wgpu object.
///
/// An `Id<T>` value identifies a value stored in a `Global`'s `Hub`.
#[repr(transparent)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Id<T: Marker>(RawId, PhantomData<T>);

// This type represents Id in a more readable (and editable) way.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub enum SerialId {
    // The only variant forces RON to not ignore "Id"
    Id(Index, Epoch),
}

impl From<RawId> for SerialId {
    fn from(id: RawId) -> Self {
        let (index, epoch) = id.unzip();
        Self::Id(index, epoch)
    }
}

pub struct ZeroIdError;

impl fmt::Display for ZeroIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IDs may not be zero")
    }
}

impl TryFrom<SerialId> for RawId {
    type Error = ZeroIdError;
    fn try_from(id: SerialId) -> Result<Self, ZeroIdError> {
        let SerialId::Id(index, epoch) = id;
        if index == 0 && epoch == 0 {
            Err(ZeroIdError)
        } else {
            Ok(RawId::zip(index, epoch))
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

    #[inline]
    pub fn zip(index: Index, epoch: Epoch) -> Self {
        Id(RawId::zip(index, epoch), PhantomData)
    }

    #[inline]
    pub fn unzip(self) -> (Index, Epoch) {
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
        let (index, epoch) = self.unzip();
        write!(formatter, "{}Id({index},{epoch})", T::TYPE)?;
        Ok(())
    }
}

impl<T> Hash for Id<T>
where
    T: Marker,
{
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
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

/// Define identifiers for each resource.
macro_rules! ids {
    ($(
        $(#[$($meta:meta)*])*
        pub type $name:ident $marker:ident;
    )*) => {
        $(
            $(#[$($meta)*])*
            pub type $name = Id<markers::$marker>;
        )*
    }
}

ids! {
    pub type AdapterId Adapter;
    pub type SurfaceId Surface;
    pub type DeviceId Device;
    pub type QueueId Queue;
    pub type BufferId Buffer;
    pub type TextureViewId TextureView;
    pub type TextureId Texture;
    pub type ExternalTextureId ExternalTexture;
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
    pub type BlasId Blas;
    pub type TlasId Tlas;
}

#[test]
fn test_id() {
    struct TestMarker;

    impl Marker for TestMarker {
        const TYPE: &'static str = "TestMarker";
    }

    let indexes = [0, Index::MAX / 2 - 1, Index::MAX / 2 + 1, Index::MAX];
    let epochs = [1, Epoch::MAX / 2 - 1, Epoch::MAX / 2 + 1, Epoch::MAX];
    for &i in &indexes {
        for &e in &epochs {
            let id = Id::<TestMarker>::zip(i, e);
            let (index, epoch) = id.unzip();
            assert_eq!(index, i);
            assert_eq!(epoch, e);
        }
    }
}
