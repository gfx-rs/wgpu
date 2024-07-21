mod command;
mod device;

pub use self::command::DynCommandEncoder;
pub use self::device::DynDevice;

use std::any::Any;

use wgt::WasmNotSendSync;

use crate::BufferBinding;

/// Base trait for all resources, allows downcasting via [`Any`].
pub trait DynResource: Any + WasmNotSendSync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Utility macro for implementing `DynResource` for a list of types.
macro_rules! impl_dyn_resource {
    ($($type:ty),*) => {
        $(
            impl crate::DynResource for $type {
                fn as_any(&self) -> &dyn ::std::any::Any {
                    self
                }

                fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                    self
                }
            }
        )*
    };
}
pub(crate) use impl_dyn_resource;

/// Extension trait for `DynResource` used by implementations of various dynamic resource traits.
trait DynResourceExt {
    /// # Panics
    ///
    /// - Panics if `self` is not downcastable to `T`.
    fn expect_downcast_ref<T: DynResource>(&self) -> &T;
    /// # Panics
    ///
    /// - Panics if `self` is not downcastable to `T`.
    fn expect_downcast_mut<T: DynResource>(&mut self) -> &mut T;

    /// Unboxes a `Box<dyn DynResource>` to a concrete type.
    ///
    /// # Safety
    ///
    /// - `self` must be the correct concrete type.
    unsafe fn unbox<T: DynResource + 'static>(self: Box<Self>) -> T;
}

impl<R: DynResource + ?Sized> DynResourceExt for R {
    fn expect_downcast_ref<'a, T: DynResource>(&'a self) -> &'a T {
        self.as_any()
            .downcast_ref()
            .expect("Resource doesn't have the expected backend type.")
    }

    fn expect_downcast_mut<'a, T: DynResource>(&'a mut self) -> &'a mut T {
        self.as_any_mut()
            .downcast_mut()
            .expect("Resource doesn't have the expected backend type.")
    }

    unsafe fn unbox<T: DynResource + 'static>(self: Box<Self>) -> T {
        debug_assert!(
            <Self as Any>::type_id(self.as_ref()) == std::any::TypeId::of::<T>(),
            "Resource doesn't have the expected type, expected {:?}, got {:?}",
            std::any::TypeId::of::<T>(),
            <Self as Any>::type_id(self.as_ref())
        );

        let casted_ptr = Box::into_raw(self).cast::<T>();
        // SAFETY: This is adheres to the safety contract of `Box::from_raw` because:
        //
        // - We are casting the value of a previously `Box`ed value, which guarantees:
        //   - `casted_ptr` is not null.
        //   - `casted_ptr` is valid for reads and writes, though by itself this does not mean
        //     valid reads and writes for `T` (read on for that).
        // - We don't change the allocator.
        // - The contract of `Box::from_raw` requires that an initialized and aligned `T` is stored
        //   within `casted_ptr`.
        *unsafe { Box::from_raw(casted_ptr) }
    }
}

pub trait DynAccelerationStructure: DynResource + std::fmt::Debug {}
pub trait DynBindGroup: DynResource + std::fmt::Debug {}
pub trait DynBindGroupLayout: DynResource + std::fmt::Debug {}
pub trait DynBuffer: DynResource + std::fmt::Debug {}
pub trait DynCommandBuffer: DynResource + std::fmt::Debug {}
pub trait DynComputePipeline: DynResource + std::fmt::Debug {}
pub trait DynFence: DynResource + std::fmt::Debug {}
pub trait DynPipelineCache: DynResource + std::fmt::Debug {}
pub trait DynPipelineLayout: DynResource + std::fmt::Debug {}
pub trait DynQuerySet: DynResource + std::fmt::Debug {}
pub trait DynRenderPipeline: DynResource + std::fmt::Debug {}
pub trait DynSampler: DynResource + std::fmt::Debug {}
pub trait DynShaderModule: DynResource + std::fmt::Debug {}
pub trait DynSurfaceTexture: DynResource + std::fmt::Debug {}
pub trait DynTexture: DynResource + std::fmt::Debug {}
pub trait DynTextureView: DynResource + std::fmt::Debug {}

impl<'a> BufferBinding<'a, dyn DynBuffer> {
    pub fn expect_downcast<B: DynBuffer>(self) -> BufferBinding<'a, B> {
        BufferBinding {
            buffer: self.buffer.expect_downcast_ref(),
            offset: self.offset,
            size: self.size,
        }
    }
}
