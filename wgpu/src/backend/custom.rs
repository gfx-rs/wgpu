//! Provides wrappers custom backend implementations

#![allow(ambiguous_wide_pointer_comparisons)]

pub use crate::dispatch::*;

use crate::cmp::AtomicU64;
use alloc::boxed::Box;
use alloc::sync::Arc;
use core::sync::atomic::Ordering;

macro_rules! dyn_type {
    // cloning of arc forbidden
    // but we still use it to provide Eq,Ord,Hash implementations
    (pub mut struct $name:ident(dyn $interface:tt)) => {
        #[derive(Debug)]
        pub(crate) struct $name(Arc<dyn $interface>);
        crate::cmp::impl_eq_ord_hash_arc_address!($name => .0);

        impl $name {
            pub(crate) fn new<T: $interface>(t: T) -> Self {
                Self(Arc::new(t))
            }

            #[allow(clippy::allow_attributes, dead_code)]
            pub(crate) fn downcast<T: $interface>(&self) -> Option<&T> {
                self.0.as_ref().as_any().downcast_ref()
            }
        }

        impl core::ops::Deref for $name {
            type Target = dyn $interface;

            #[inline]
            fn deref(&self) -> &Self::Target {
                self.0.as_ref()
            }
        }

        impl core::ops::DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                Arc::get_mut(&mut self.0).expect("")
            }
        }
    };
    // cloning of arc is allowed
    (pub ref struct $name:ident(dyn $interface:tt)) => {
        #[derive(Debug, Clone)]
        pub(crate) struct $name(Arc<dyn $interface>);
        crate::cmp::impl_eq_ord_hash_arc_address!($name => .0);

        impl $name {
            pub(crate) fn new<T: $interface>(t: T) -> Self {
                Self(Arc::new(t))
            }

            pub(crate) fn downcast<T: $interface>(&self) -> Option<&T> {
                self.0.as_ref().as_any().downcast_ref()
            }
        }

        impl core::ops::Deref for $name {
            type Target = dyn $interface;

            #[inline]
            fn deref(&self) -> &Self::Target {
                self.0.as_ref()
            }
        }
    };
}

dyn_type!(pub ref struct DynContext(dyn InstanceInterface));
dyn_type!(pub ref struct DynAdapter(dyn AdapterInterface));
dyn_type!(pub ref struct DynDevice(dyn DeviceInterface));
dyn_type!(pub ref struct DynQueue(dyn QueueInterface));
dyn_type!(pub ref struct DynShaderModule(dyn ShaderModuleInterface));
dyn_type!(pub ref struct DynBindGroupLayout(dyn BindGroupLayoutInterface));
dyn_type!(pub ref struct DynBindGroup(dyn BindGroupInterface));
dyn_type!(pub ref struct DynTextureView(dyn TextureViewInterface));
dyn_type!(pub ref struct DynSampler(dyn SamplerInterface));
dyn_type!(pub ref struct DynBuffer(dyn BufferInterface));
dyn_type!(pub ref struct DynTexture(dyn TextureInterface));
dyn_type!(pub ref struct DynExternalTexture(dyn ExternalTextureInterface));
dyn_type!(pub ref struct DynBlas(dyn BlasInterface));
dyn_type!(pub ref struct DynTlas(dyn TlasInterface));
dyn_type!(pub ref struct DynQuerySet(dyn QuerySetInterface));
dyn_type!(pub ref struct DynPipelineLayout(dyn PipelineLayoutInterface));
dyn_type!(pub ref struct DynRenderPipeline(dyn RenderPipelineInterface));
dyn_type!(pub ref struct DynComputePipeline(dyn ComputePipelineInterface));
dyn_type!(pub ref struct DynPipelineCache(dyn PipelineCacheInterface));
dyn_type!(pub mut struct DynCommandEncoder(dyn CommandEncoderInterface));
dyn_type!(pub mut struct DynComputePass(dyn ComputePassInterface));
dyn_type!(pub mut struct DynRenderPass(dyn RenderPassInterface));
dyn_type!(pub mut struct DynCommandBuffer(dyn CommandBufferInterface));

static NEXT_RENDER_BUNDLE_ENCODER_ID: AtomicU64 = AtomicU64::new(0);

// DynRenderBundleEncoder uses Box instead of Arc so that finish_boxed(self: Box<Self>)
// can be dispatched through the trait object (consuming the encoder).
#[derive(Debug)]
pub(crate) struct DynRenderBundleEncoder {
    // Unique identity for Eq/Ord/Hash. The data pointer of the boxed trait object is
    // not safe to use for identity because ZST impls share the same dangling address.
    id: u64,
    inner: Box<dyn RenderBundleEncoderInterface>,
}

impl DynRenderBundleEncoder {
    pub(crate) fn new<T: RenderBundleEncoderInterface>(t: T) -> Self {
        Self {
            id: NEXT_RENDER_BUNDLE_ENCODER_ID.fetch_add(1, Ordering::Relaxed),
            inner: Box::new(t),
        }
    }

    pub(crate) fn downcast<T: RenderBundleEncoderInterface>(&self) -> Option<&T> {
        self.inner.as_ref().as_any().downcast_ref()
    }

    pub(crate) fn finish_boxed(
        self,
        desc: &crate::RenderBundleDescriptor<'_>,
    ) -> crate::dispatch::DispatchRenderBundle {
        self.inner.finish_boxed(desc)
    }
}

impl core::ops::Deref for DynRenderBundleEncoder {
    type Target = dyn RenderBundleEncoderInterface;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

impl core::ops::DerefMut for DynRenderBundleEncoder {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut()
    }
}

// Eq/Ord/Hash for DynRenderBundleEncoder are based on a unique id assigned at construction.
//
// These impls are not semantically meaningful (we never sort or deduplicate encoders by
// "value") but are required to satisfy bounds imposed by the dispatch enum machinery.
impl PartialEq for DynRenderBundleEncoder {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for DynRenderBundleEncoder {}

impl PartialOrd for DynRenderBundleEncoder {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DynRenderBundleEncoder {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl core::hash::Hash for DynRenderBundleEncoder {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

dyn_type!(pub ref struct DynRenderBundle(dyn RenderBundleInterface));
dyn_type!(pub ref struct DynSurface(dyn SurfaceInterface));
dyn_type!(pub ref struct DynSurfaceOutputDetail(dyn SurfaceOutputDetailInterface));
dyn_type!(pub mut struct DynQueueWriteBuffer(dyn QueueWriteBufferInterface));
dyn_type!(pub mut struct DynBufferMappedRange(dyn BufferMappedRangeInterface));
