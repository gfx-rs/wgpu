//! Reflection information about a validated shader module.
//!
//! These types describe a module's interface without referring to naga's IR arenas,
//! so they can be produced by naga, serialized, and consumed by code that does not
//! depend on naga.

use alloc::{string::String, vec::Vec};
use core::num::NonZeroU32;

use crate::{
    AddressSpace, BuiltIn, ImageClass, ImageDimension, ImmediateUsage, Interpolation,
    MeshOutputTopology, ResourceBinding, Sampling, Scalar, ShaderStage, VectorSize,
};

/// Index into [`ModuleReflection::types`].
pub type TypeIndex = usize;

/// Index into [`ModuleReflection::resources`].
pub type ResourceIndex = usize;

/// Index into [`ModuleReflection::overrides`].
pub type OverrideIndex = usize;

/// Reflection information for a whole module.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct ModuleReflection {
    /// Every type in the module, in the order of the module's type arena.
    pub types: Vec<Type>,
    /// Every global variable with a `@group`/`@binding` attribute.
    pub resources: Vec<Resource>,
    /// Every pipeline-overridable constant.
    pub overrides: Vec<Override>,
    /// Every entry point.
    pub entry_points: Vec<EntryPoint>,
}

impl ModuleReflection {
    /// Find the entry point with the given stage and name.
    pub fn entry_point(&self, stage: ShaderStage, name: &str) -> Option<&EntryPoint> {
        self.entry_points
            .iter()
            .find(|ep| ep.stage == stage && ep.name == name)
    }
}

/// A type in the module.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct Type {
    /// The name of the type, if it is a named struct or alias.
    pub name: Option<String>,
    /// Size in bytes. For runtime-sized arrays, and structs ending in one, this
    /// assumes a single element.
    pub size: u32,
    /// Alignment in bytes.
    pub align: u32,
    pub inner: TypeInner,
}

/// The shape of a [`Type`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase", rename_all_fields = "camelCase")
)]
pub enum TypeInner {
    Scalar(Scalar),
    Vector {
        size: VectorSize,
        scalar: Scalar,
    },
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
    },
    Atomic(Scalar),
    Pointer {
        base: TypeIndex,
        space: AddressSpace,
    },
    Array {
        base: TypeIndex,
        size: ArraySize,
        stride: u32,
    },
    Struct {
        members: Vec<StructMember>,
        span: u32,
    },
    Image {
        dim: ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
    Sampler {
        comparison: bool,
    },
    AccelerationStructure {
        vertex_return: bool,
    },
    RayQuery {
        vertex_return: bool,
    },
    BindingArray {
        base: TypeIndex,
        size: ArraySize,
    },
    /// A type that reflection does not describe in detail.
    Other,
}

/// The element count of an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub enum ArraySize {
    Constant(NonZeroU32),
    /// The count is given by an override.
    Override(OverrideIndex),
    /// The count is determined by the bound buffer at runtime.
    Dynamic,
}

/// A member of a struct type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct StructMember {
    pub name: Option<String>,
    pub ty: TypeIndex,
    pub offset: u32,
}

/// A bound resource: a global variable with `@group` and `@binding`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct Resource {
    pub name: Option<String>,
    pub binding: ResourceBinding,
    pub space: AddressSpace,
    /// The declared type of the variable.
    pub ty: TypeIndex,
    /// What kind of resource this is. For binding arrays, this describes the element.
    pub kind: ResourceKind,
    /// The element count, if this is a binding array.
    pub binding_array_size: Option<ArraySize>,
}

/// The kind of a [`Resource`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase", rename_all_fields = "camelCase")
)]
pub enum ResourceKind {
    Buffer {
        /// The minimum size of a buffer binding for this resource, in bytes.
        min_binding_size: NonZeroU32,
    },
    Texture {
        dim: ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
    Sampler {
        comparison: bool,
    },
    AccelerationStructure {
        vertex_return: bool,
    },
}

/// A pipeline-overridable constant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct Override {
    pub name: Option<String>,
    /// The numeric `@id`, if any.
    pub id: Option<u16>,
    pub ty: TypeIndex,
    /// Whether the override has an initializer. Overrides without one must be given a
    /// value when creating a pipeline.
    pub has_default: bool,
}

/// The type of a user-defined or built-in stage input or output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase", rename_all_fields = "camelCase")
)]
pub enum NumericType {
    Scalar(Scalar),
    Vector {
        size: VectorSize,
        scalar: Scalar,
    },
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        scalar: Scalar,
    },
}

/// A stage input or output, after flattening any structs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase", rename_all_fields = "camelCase")
)]
pub enum Varying {
    /// A value with a `@location` attribute.
    Location {
        name: Option<String>,
        location: u32,
        ty: NumericType,
        interpolation: Option<Interpolation>,
        sampling: Option<Sampling>,
        per_primitive: bool,
        blend_src: Option<u32>,
    },
    /// A value with a `@builtin` attribute.
    BuiltIn {
        name: Option<String>,
        built_in: BuiltIn,
        /// The type, if it is numeric. `None` for arrays such as `clip_distances`.
        ty: Option<NumericType>,
        /// The element count, for array-typed built-ins such as `clip_distances`.
        array_size: Option<u32>,
    },
}

/// Information specific to mesh shader entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct MeshInfo {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub topology: MeshOutputTopology,
}

/// Reflection information for one entry point.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(rename_all = "camelCase")
)]
pub struct EntryPoint {
    pub name: String,
    pub stage: ShaderStage,
    /// Stage inputs, with structs flattened.
    pub inputs: Vec<Varying>,
    /// Stage outputs, with structs flattened. For mesh shaders, this also holds the
    /// per-vertex and per-primitive outputs.
    pub outputs: Vec<Varying>,
    /// The resources this entry point statically uses.
    pub resources: Vec<ResourceIndex>,
    /// (texture, sampler) pairs this entry point samples with.
    pub sampling_pairs: Vec<(ResourceIndex, ResourceIndex)>,
    /// The workgroup size, for compute-like stages. `[0, 0, 0]` for other stages.
    pub workgroup_size: [u32; 3],
    /// For each dimension, whether the workgroup size is given by an override
    /// expression. Such dimensions hold a placeholder in [`Self::workgroup_size`] until
    /// overrides are applied.
    pub workgroup_size_from_overrides: [bool; 3],
    /// Bytes of `workgroup` memory used, with each variable rounded up to 16 bytes.
    pub workgroup_storage_size: u64,
    pub dual_source_blending: bool,
    /// Size of the task payload, for task and mesh shaders that use one.
    pub task_payload_size: Option<u32>,
    pub mesh_info: Option<MeshInfo>,
    /// The immediate data this entry point uses.
    pub immediate_usage: ImmediateUsage,
    /// The type of the ray tracing payload, for ray tracing stages.
    pub ray_tracing_payload: Option<TypeIndex>,
}
