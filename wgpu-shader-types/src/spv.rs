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
