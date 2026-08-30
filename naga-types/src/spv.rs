#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindingInfo {
    pub descriptor_set: u32,
    pub binding: u32,
    /// If the binding is an unsized binding array, this overrides the size.
    pub binding_array_size: Option<u32>,
}

// Using `BTreeMap` instead of `HashMap` so that we can hash itself.
pub type BindingMap = alloc::collections::BTreeMap<crate::ResourceBinding, BindingInfo>;

/// The descriptor set and binding at which the SPIR-V backend places the
/// resource table's synthesized descriptor array(s).
///
/// A module that uses `getResource<T>` has no Naga `GlobalVariable` for the
/// table: the backend synthesizes one `OpVariable` of type
/// `OpTypeRuntimeArray<T>` per distinct resource type `T`, and decorates every
/// one of them with *this same* descriptor set and binding (descriptor
/// aliasing). The consuming backend (wgpu-hal) must therefore bind the table's
/// descriptor set at `descriptor_set`, with the image array at `binding`.
///
/// This is a struct rather than a bare `(u32, u32)` tuple so that later
/// milestones can add the metadata-buffer and mask-ring bindings without
/// breaking this option field.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct ResourceTableBindTarget {
    /// The descriptor set the table's descriptor array(s) bind to.
    pub descriptor_set: u32,
    /// The binding within `descriptor_set` for the image array.
    pub binding: u32,
}
