#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::Features;

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a [resource table][Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE].
pub struct ResourceTableDescriptor<L> {
    /// Label for the resource table.
    pub label: L,
    /// Number of slots in the table.
    ///
    /// Must be less than or equal to 65536.
    ///
    /// The descriptor array backing the table is sparse: slots start out empty and are
    /// populated (and later cleared) over time via `update`/`insert_binding`.
    pub size: u32,
}

impl<L> ResourceTableDescriptor<L> {
    /// Takes a closure and maps the label of the resource table descriptor into another.
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> ResourceTableDescriptor<K> {
        ResourceTableDescriptor {
            label: fun(&self.label),
            size: self.size,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// The usage a resource has been given within a resource table, controlling its visibility to
/// `getResource`/`hasResource` accesses of that table.
pub enum ResourceTableUsage {
    /// The resource is hidden from resource table access: it behaves as if not present in the
    /// table for the purposes of `getResource`/`hasResource`.
    None,
    /// The resource is visible for read-only resource table access.
    ///
    /// This is the default usage for resources newly added to a table.
    #[default]
    ReadOnly,
    /// The resource is visible for writable resource table access.
    ///
    /// Requires [`Features::EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE`]; unused until later
    /// milestones of the resource table implementation.
    Writable,
}
