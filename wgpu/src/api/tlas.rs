use crate::{api::blas::TlasInstance, dispatch};
use crate::{BindingResource, Buffer, Label};
use std::ops::{Index, IndexMut, Range};
use wgt::WasmNotSendSync;

/// Descriptor to create top level acceleration structures.
pub type CreateTlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CreateTlasDescriptor<'_>: Send, Sync);

#[derive(Debug)]
/// Top Level Acceleration Structure (TLAS).
///
/// A TLAS contains a series of [TLAS instances], which are a reference to
/// a BLAS and a transformation matrix placing the geometry in the world.
///
/// A TLAS contains TLAS instances in a device readable form, you cant interact
/// directly with these, instead you have to build the TLAS with [TLAS instances].
///
/// [TLAS instances]: TlasInstance
pub struct Tlas {
    pub(crate) inner: dispatch::DispatchTlas,
    pub(crate) max_instances: u32,
}
static_assertions::assert_impl_all!(Tlas: WasmNotSendSync);

crate::cmp::impl_eq_ord_hash_proxy!(Tlas => .inner);

impl Tlas {
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        self.inner.destroy();
    }
}

/// Entry for a top level acceleration structure build.
/// Used with raw instance buffers for an unvalidated builds.
/// See [TlasPackage] for the safe version.
pub struct TlasBuildEntry<'a> {
    /// Reference to the acceleration structure.
    pub tlas: &'a Tlas,
    /// Reference to the raw instance buffer, each instance is similar to [TlasInstance] but contains a handle to the BLAS.
    pub instance_buffer: &'a Buffer,
    /// Number of instances in the instance buffer.
    pub instance_count: u32,
}
static_assertions::assert_impl_all!(TlasBuildEntry<'_>: WasmNotSendSync);

/// The safe version of TlasEntry, containing TlasInstances instead of a raw buffer.
pub struct TlasPackage {
    pub(crate) tlas: Tlas,
    pub(crate) instances: Vec<Option<TlasInstance>>,
    pub(crate) lowest_unmodified: u32,
}
static_assertions::assert_impl_all!(TlasPackage: WasmNotSendSync);

impl TlasPackage {
    /// Construct [TlasPackage] consuming the [Tlas] (prevents modification of the [Tlas] without using this package).
    pub fn new(tlas: Tlas) -> Self {
        let max_instances = tlas.max_instances;
        Self::new_with_instances(tlas, vec![None; max_instances as usize])
    }

    /// Construct [TlasPackage] consuming the [Tlas] (prevents modification of the Tlas without using this package).
    /// This constructor moves the instances into the package (the number of instances needs to fit into tlas,
    /// otherwise when building a validation error will be raised).
    pub fn new_with_instances(tlas: Tlas, instances: Vec<Option<TlasInstance>>) -> Self {
        Self {
            tlas,
            lowest_unmodified: instances.len() as u32,
            instances,
        }
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
    /// to use [Self::index_mut] unless the option if out of bounds is required
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
    /// to use [Self::index_mut] unless the option if out of bounds is required
    pub fn get_mut_single(&mut self, index: usize) -> Option<&mut Option<TlasInstance>> {
        if index >= self.instances.len() {
            return None;
        }
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        Some(&mut self.instances[index])
    }

    /// Get the binding resource for the underling acceleration structure, to be used when creating a [BindGroup]
    ///
    /// [BindGroup]: super::BindGroup
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(&self.tlas)
    }

    /// Get a reference to the underling [Tlas].
    pub fn tlas(&self) -> &Tlas {
        &self.tlas
    }
}

impl Index<usize> for TlasPackage {
    type Output = Option<TlasInstance>;

    fn index(&self, index: usize) -> &Self::Output {
        self.instances.index(index)
    }
}

impl Index<Range<usize>> for TlasPackage {
    type Output = [Option<TlasInstance>];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        self.instances.index(index)
    }
}

impl IndexMut<usize> for TlasPackage {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = self.instances.index_mut(index);
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        idx
    }
}

impl IndexMut<Range<usize>> for TlasPackage {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        let idx = self.instances.index_mut(index.clone());
        if index.end > self.lowest_unmodified as usize {
            self.lowest_unmodified = index.end as u32;
        }
        idx
    }
}
