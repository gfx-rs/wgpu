use crate::{api::blas::TlasInstance, dispatch};
use crate::{BindingResource, Label};
use alloc::vec::Vec;
use core::ops::{Index, IndexMut, Range};
use wgt::WasmNotSendSync;

/// Descriptor to create top level acceleration structures.
pub type CreateTlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CreateTlasDescriptor<'_>: Send, Sync);

#[derive(Debug, Clone)]
/// Top Level Acceleration Structure (TLAS).
///
/// A TLAS contains a series of [TLAS instances], which are a reference to
/// a BLAS and a transformation matrix placing the geometry in the world.
///
/// A TLAS also contains an extra set of TLAS instances in a device readable form, you cant interact
/// directly with these, instead you have to build the TLAS with [TLAS instances].
///
/// [TLAS instances]: TlasInstance
pub struct Tlas {
    pub(crate) inner: dispatch::DispatchTlas,
    pub(crate) instances: Vec<Option<TlasInstance>>,
    pub(crate) lowest_unmodified: u32,
}
static_assertions::assert_impl_all!(Tlas: WasmNotSendSync);

crate::cmp::impl_eq_ord_hash_proxy!(Tlas => .inner);

impl Tlas {
    /// Returns the inner hal Acceleration Structure using a callback. The hal acceleration structure
    /// will be `None` if the backend type argument does not match with this wgpu Tlas
    ///
    /// This method will start the wgpu_core level command recording.
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Acceleration Structure must not be manually destroyed
    /// - If the raw handle is build,
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::AccelerationStructure>) -> R,
        R,
    >(
        &mut self,
        hal_tlas_callback: F,
    ) -> R {
        if let Some(tlas) = self.inner.as_core_opt() {
            unsafe { tlas.context.tlas_as_hal::<A, F, R>(tlas, hal_tlas_callback) }
        } else {
            hal_tlas_callback(None)
        }
    }

    #[cfg(custom)]
    /// Returns custom implementation of Tlas (if custom backend and is internally T)
    pub fn as_custom<T: crate::custom::TlasInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }

    /// Get a reference to all instances.
    pub fn get(&self) -> &[Option<TlasInstance>] {
        &self.instances
    }

    /// Get a mutable slice to a range of instances.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used. It is recommended
    /// to use [`Self::index_mut`] unless the option if out of bounds is required
    pub fn get_mut_slice(&mut self, range: Range<usize>) -> Option<&mut [Option<TlasInstance>]> {
        if range.end > self.instances.len() {
            return None;
        }
        if range.end as u32 > self.lowest_unmodified {
            self.lowest_unmodified = range.end as u32;
        }
        Some(&mut self.instances[range])
    }

    /// Get a single mutable reference to an instance.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used. It is recommended
    /// to use [`Self::index_mut`] unless the option if out of bounds is required
    pub fn get_mut_single(&mut self, index: usize) -> Option<&mut Option<TlasInstance>> {
        if index >= self.instances.len() {
            return None;
        }
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        Some(&mut self.instances[index])
    }

    /// Get the binding resource for the underling acceleration structure, to be used when creating a [`BindGroup`]
    ///
    /// [`BindGroup`]: super::BindGroup
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(self)
    }
}

impl Index<usize> for Tlas {
    type Output = Option<TlasInstance>;

    fn index(&self, index: usize) -> &Self::Output {
        self.instances.index(index)
    }
}

impl Index<Range<usize>> for Tlas {
    type Output = [Option<TlasInstance>];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        self.instances.index(index)
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = self.instances.index_mut(index);
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        idx
    }
}

impl IndexMut<Range<usize>> for Tlas {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        let idx = self.instances.index_mut(index.clone());
        if index.end > self.lowest_unmodified as usize {
            self.lowest_unmodified = index.end as u32;
        }
        idx
    }
}
