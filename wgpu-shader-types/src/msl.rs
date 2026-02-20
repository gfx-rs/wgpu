use alloc::string::String;
use alloc::vec::Vec;

pub type Slot = u8;
pub type InlineSamplerIndex = u8;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum BindSamplerTarget {
    Resource(Slot),
    Inline(InlineSamplerIndex),
}

/// Binding information for a Naga [`External`] image global variable.
///
/// See the module documentation's section on external textures for details.
///
/// [`External`]: crate::ir::ImageClass::External
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindExternalTextureTarget {
    pub planes: [Slot; 3],
    pub params: Slot,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(any(feature = "serialize", feature = "deserialize"), serde(default))]
pub struct BindTarget {
    pub buffer: Option<Slot>,
    pub texture: Option<Slot>,
    pub sampler: Option<BindSamplerTarget>,
    pub external_texture: Option<BindExternalTextureTarget>,
    pub mutable: bool,
}

// Using `BTreeMap` instead of `HashMap` so that we can hash itself.
pub type BindingMap = alloc::collections::BTreeMap<crate::ResourceBinding, BindTarget>;

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(any(feature = "serialize", feature = "deserialize"), serde(default))]
pub struct EntryPointResources {
    #[cfg_attr(
        feature = "deserialize",
        serde(deserialize_with = "deserialize_binding_map")
    )]
    pub resources: BindingMap,

    pub immediates_buffer: Option<Slot>,

    /// The slot of a buffer that contains an array of `u32`,
    /// one for the size of each bound buffer that contains a runtime array,
    /// in order of [`crate::GlobalVariable`] declarations.
    pub sizes_buffer: Option<Slot>,
}

pub type EntryPointResourceMap = alloc::collections::BTreeMap<String, EntryPointResources>;

#[cfg(feature = "deserialize")]
#[derive(serde::Deserialize)]
struct BindingMapSerialization {
    resource_binding: crate::ResourceBinding,
    bind_target: BindTarget,
}

#[cfg(feature = "deserialize")]
fn deserialize_binding_map<'de, D>(deserializer: D) -> Result<BindingMap, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;

    let vec = Vec::<BindingMapSerialization>::deserialize(deserializer)?;
    let mut map = BindingMap::default();
    for item in vec {
        map.insert(item.resource_binding, item.bind_target);
    }
    Ok(map)
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum EntryPointError {
    #[error("global '{0}' doesn't have a binding")]
    MissingBinding(String),
    #[error("mapping of {0:?} is missing")]
    MissingBindTarget(crate::ResourceBinding),
    #[error("mapping for immediates is missing")]
    MissingImmediateData,
    #[error("mapping for sizes buffer is missing")]
    MissingSizesBuffer,
}

/// Information about a translated module that is required
/// for the use of the result.
pub struct TranslationInfo {
    /// Mapping of the entry point names. Each item in the array
    /// corresponds to an entry point index.
    ///
    ///Note: Some entry points may fail translation because of missing bindings.
    pub entry_point_names: Vec<Result<String, EntryPointError>>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct VertexBufferMapping {
    pub id: u32,
    pub stride: u32,
    pub step_mode: crate::VertexStepMode,
    pub attributes: Vec<crate::VertexAttribute>,
}
