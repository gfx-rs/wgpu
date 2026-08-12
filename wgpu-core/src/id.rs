use wgt::WasmNotSendSync;

/// Identify an object by the pointer returned by `Arc::as_ptr`.
///
/// This is used for tracing.
///
/// As of `wgpu` v27, commands are encoded all at once when
/// `CommandEncoder::finish` is called, not when the encoding methods are
/// called for each command. This implies storing a representation of the
/// commands in memory until `finish` is called. The
/// serialized trace identifies resources by the integer value of
/// `Arc::as_ptr`. These IDs have the type [`crate::id::PointerId`]. The
/// trace player uses hash maps to go from `PointerId`s to `Arc`s
/// when replaying a trace.
#[allow(dead_code)]
#[cfg(feature = "serde")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum PointerId<T: Marker> {
    // The only variant forces RON to not ignore "Id"
    PointerId(
        core::num::NonZeroUsize,
        #[serde(skip)] core::marker::PhantomData<T>,
    ),
}

#[cfg(feature = "serde")]
impl<T: Marker> Copy for PointerId<T> {}

#[cfg(feature = "serde")]
impl<T: Marker> Clone for PointerId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "serde")]
impl<T: Marker> PartialEq for PointerId<T> {
    fn eq(&self, other: &Self) -> bool {
        let PointerId::PointerId(this, _) = self;
        let PointerId::PointerId(other, _) = other;
        this == other
    }
}

#[cfg(feature = "serde")]
impl<T: Marker> Eq for PointerId<T> {}

#[cfg(feature = "serde")]
impl<T: Marker> core::hash::Hash for PointerId<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        let PointerId::PointerId(this, _) = self;
        this.hash(state);
    }
}

#[cfg(feature = "serde")]
impl<T: crate::storage::StorageItem> From<&alloc::sync::Arc<T>> for PointerId<T::Marker> {
    fn from(arc: &alloc::sync::Arc<T>) -> Self {
        // Since the memory representation of `Arc<T>` is just a pointer to
        // `ArcInner<T>`, it would be nice to use that pointer as the trace ID,
        // since many `into_trace` implementations would then be no-ops at
        // runtime. However, `Arc::as_ptr` returns a pointer to the contained
        // data, not to the `ArcInner`. The `ArcInner` stores the reference
        // counts before the data, so the machine code for this conversion has
        // to add an offset to the pointer.
        PointerId::PointerId(
            core::num::NonZeroUsize::new(alloc::sync::Arc::as_ptr(arc) as usize).unwrap(),
            core::marker::PhantomData,
        )
    }
}

/// Marker trait used to determine which types uniquely identify a resource.
///
/// For example, `Device<A>` will have the same type of identifier as
/// `Device<B>` because `Device<T>` for any `T` defines the same maker type.
pub trait Marker: 'static + WasmNotSendSync {
    const TYPE: &'static str;
}

// This allows `()` to be used as a marker type for tests.
//
// We don't want these in production code, since they essentially remove type
// safety, like how identifiers across different types can be compared.
#[cfg(test)]
impl Marker for () {
    const TYPE: &'static str = "Untyped";
}

/// Define markers for each resource.
macro_rules! ids {
    ($(
        $(#[$($meta:meta)*])*
        pub type $marker:ident;
    )*) => {
        /// Marker types for each resource.
        pub mod markers {
            $(
                #[derive(Debug)]
                pub enum $marker {}
                impl super::Marker for $marker {
                    const TYPE: &'static str = stringify!($marker);
                }
            )*
        }
    }
}

ids! {
    pub type Adapter;
    pub type Surface;
    pub type Device;
    pub type Queue;
    pub type Buffer;
    pub type StagingBuffer;
    pub type TextureView;
    pub type Texture;
    pub type ExternalTexture;
    pub type Sampler;
    pub type BindGroupLayout;
    pub type PipelineLayout;
    pub type BindGroup;
    pub type ShaderModule;
    pub type RenderPipeline;
    pub type ComputePipeline;
    pub type PipelineCache;
    pub type CommandEncoder;
    pub type CommandBuffer;
    pub type RenderPassEncoder;
    pub type ComputePassEncoder;
    pub type RenderBundleEncoder;
    pub type RenderBundle;
    pub type QuerySet;
    pub type Blas;
    pub type Tlas;
}
