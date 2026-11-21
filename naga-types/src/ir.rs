//! Shader IR types shared between naga and its consumers.
//!
//! These are re-exported from `naga` and are part of its IR.

/// Addressing space of variables.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AddressSpace {
    /// Function locals.
    Function,
    /// Private data, per invocation, mutable.
    Private,
    /// Workgroup shared data, mutable.
    WorkGroup,
    /// Uniform buffer data.
    Uniform,
    /// Storage buffer data, potentially mutable.
    Storage { access: StorageAccess },
    /// Opaque handles, such as samplers and images.
    Handle,

    /// Immediate data.
    ///
    /// A `naga::Module` may contain at most one `naga::GlobalVariable` in
    /// this address space. Its contents are provided not by a buffer
    /// but by `SetImmediates` pass commands, allowing the CPU to
    /// establish different values for each draw/dispatch.
    ///
    /// `Immediate` variables may not contain `f16` values, even if
    /// the `SHADER_FLOAT16` capability is enabled.
    ///
    /// Backends generally place tight limits on the size of
    /// `Immediate` variables.
    Immediate,
    /// Task shader to mesh shader payload
    TaskPayload,

    /// Ray tracing payload, for inputting in TraceRays
    RayPayload,
    /// Ray tracing payload, for entrypoints invoked by a TraceRays call
    ///
    /// Each entrypoint may reference only one variable in this scope, as
    /// only one may be passed as a payload.
    IncomingRayPayload,
}

/// Built-in inputs and outputs.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum BuiltIn {
    // This must be at the top so that it gets sorted to the top. PrimitiveIndex is considered a non SV
    // by FXC so it must appear before any other SVs.
    /// Read in fragment shaders, written in mesh shaders, read in any and closest hit shaders.
    PrimitiveIndex,

    /// Written in vertex/mesh shaders, read in fragment shaders
    Position { invariant: bool },
    /// Read in task, mesh, vertex, and fragment shaders
    ViewIndex,

    /// Read in vertex shaders
    BaseInstance,
    /// Read in vertex shaders
    BaseVertex,
    /// Written in vertex & mesh shaders
    ClipDistances,
    /// Written in vertex & mesh shaders
    CullDistance,
    /// Read in vertex, any- and closest-hit shaders
    InstanceIndex,
    /// Written in vertex & mesh shaders
    PointSize,
    /// Read in vertex shaders
    VertexIndex,
    /// Read in vertex & task shaders, or mesh shaders in pipelines without task shaders
    DrawIndex,

    /// Written in fragment shaders
    FragDepth,
    /// Read in fragment shaders
    PointCoord,
    /// Read in fragment shaders
    FrontFacing,
    /// Read in fragment shaders
    Barycentric { perspective: bool },
    /// Read in fragment shaders
    SampleIndex,
    /// Read or written in fragment shaders
    SampleMask,

    /// Read in compute, task, and mesh shaders
    GlobalInvocationId,
    /// Read in compute, task, and mesh shaders
    LocalInvocationId,
    /// Read in compute, task, and mesh shaders
    LocalInvocationIndex,
    /// Read in compute, task, and mesh shaders
    WorkGroupId,
    /// Read in compute, task, and mesh shaders
    WorkGroupSize,
    /// Read in compute, task, and mesh shaders
    NumWorkGroups,

    /// Read in compute, task, and mesh shaders
    NumSubgroups,
    /// Read in compute, task, and mesh shaders
    SubgroupId,
    /// Read in compute, fragment, task, and mesh shaders
    SubgroupSize,
    /// Read in compute, fragment, task, and mesh shaders
    SubgroupInvocationId,

    /// Written in task shaders
    MeshTaskSize,
    /// Written in mesh shaders
    CullPrimitive,
    /// Written in mesh shaders
    PointIndex,
    /// Written in mesh shaders
    LineIndices,
    /// Written in mesh shaders
    TriangleIndices,

    /// Written to a workgroup variable in mesh shaders
    VertexCount,
    /// Written to a workgroup variable in mesh shaders
    Vertices,
    /// Written to a workgroup variable in mesh shaders
    PrimitiveCount,
    /// Written to a workgroup variable in mesh shaders
    Primitives,

    /// Read in all ray tracing pipeline shaders, the id within the number of
    /// rays that this current ray is.
    RayInvocationId,
    /// Read in all ray tracing pipeline shaders, the number of rays created.
    NumRayInvocations,
    /// Read in closest hit and any hit shaders, the custom data in the tlas
    /// instance
    InstanceCustomData,
    /// Read in closest hit and any hit shaders, the index of the geometry in
    /// the blas.
    GeometryIndex,
    /// Read in closest hit, any hit, and miss shaders, the origin of the ray.
    WorldRayOrigin,
    /// Read in closest hit, any hit, and miss shaders, the direction of the
    /// ray.
    WorldRayDirection,
    /// Read in closest hit and any hit shaders, the direction of the ray in
    /// object space.
    ObjectRayOrigin,
    /// Read in closest hit and any hit shaders, the direction of the ray in
    /// object space.
    ObjectRayDirection,
    /// Read in closest hit, any hit, and miss shaders, the t min provided by
    /// in the ray desc.
    RayTmin,
    /// Read in closest hit, any hit, and miss shaders, the final bounds at which
    /// a hit is accepted (the closest committed hit if there is one otherwise, t
    /// max provided in the ray desc).
    RayTCurrentMax,
    /// Read in closest hit and any hit shaders, the matrix for converting from
    /// object space to world space
    ObjectToWorld,
    /// Read in closest hit and any hit shaders, the matrix for converting from
    /// world space to object space
    WorldToObject,
    /// Read in closest hit and any hit shaders, the type of hit as provided by
    /// the intersection function if any, otherwise this is 254 (0xFE) for a
    /// front facing triangle and 255 (0xFF) for a back facing triangle
    HitKind,
    /// Read in closest hit and any hit shaders, the second and third barycentric
    /// coordinates of the hit point on the triangle. The first can be computed as
    /// `1.0 - x - y`. Only meaningful if the hit was a triangle.
    HitBarycentrics,
}

/// Number of bytes per scalar.
pub type Bytes = u8;

/// Number of components in a vector.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum VectorSize {
    /// 2D vector
    Bi = 2,
    /// 3D vector
    Tri = 3,
    /// 4D vector
    Quad = 4,
}

impl VectorSize {
    pub const MAX: usize = Self::Quad as usize;
}

impl From<VectorSize> for u8 {
    fn from(size: VectorSize) -> u8 {
        size as u8
    }
}

impl From<VectorSize> for u32 {
    fn from(size: VectorSize) -> u32 {
        size as u32
    }
}

/// Primitive type for a scalar.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ScalarKind {
    /// Signed integer type.
    Sint,
    /// Unsigned integer type.
    Uint,
    /// Floating point type.
    Float,
    /// Boolean type.
    Bool,

    /// WGSL abstract integer type.
    ///
    /// These are forbidden by validation, and should never reach backends.
    AbstractInt,

    /// Abstract floating-point type.
    ///
    /// These are forbidden by validation, and should never reach backends.
    AbstractFloat,
}

/// Characteristics of a scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct Scalar {
    /// How the value's bits are to be interpreted.
    pub kind: ScalarKind,

    /// This size of the value in bytes.
    pub width: Bytes,
}

/// The interpolation qualifier of a binding or struct field.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum Interpolation {
    /// The value will be interpolated in a perspective-correct fashion.
    /// Also known as "smooth" in glsl.
    Perspective,
    /// Indicates that linear, non-perspective, correct
    /// interpolation must be used.
    /// Also known as "no_perspective" in glsl.
    Linear,
    /// Indicates that no interpolation will be performed.
    Flat,
    /// Indicates the fragment input binding holds an array of per-vertex values.
    /// This is typically used with barycentrics.
    PerVertex,
}

/// The sampling qualifiers of a binding or struct field.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum Sampling {
    /// Interpolate the value at the center of the pixel.
    Center,

    /// Interpolate the value at a point that lies within all samples covered by
    /// the fragment within the current primitive. In multisampling, use a
    /// single value for all samples in the primitive.
    Centroid,

    /// Interpolate the value at each sample location. In multisampling, invoke
    /// the fragment shader once per sample.
    Sample,

    /// Use the value provided by the first vertex of the current primitive.
    First,

    /// Use the value provided by the first or last vertex of the current primitive. The exact
    /// choice is implementation-dependent.
    Either,
}

/// The number of dimensions an image has.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ImageDimension {
    /// 1D image
    D1,
    /// 2D image
    D2,
    /// 3D image
    D3,
    /// Cube map
    Cube,
}

bitflags::bitflags! {
    /// Flags describing an image.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    #[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct StorageAccess: u32 {
        /// Storage can be used as a source for load ops.
        const LOAD = 0x1;
        /// Storage can be used as a target for store ops.
        const STORE = 0x2;
        /// Storage can be used as a target for atomic ops.
        const ATOMIC = 0x4;
    }
}

/// Image storage format.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum StorageFormat {
    // 8-bit formats
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,

    // 16-bit formats
    R16Uint,
    R16Sint,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,

    // 32-bit formats
    R32Uint,
    R32Sint,
    R32Float,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,

    // Packed 32-bit formats
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,

    // 64-bit formats
    R64Uint,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,

    // 128-bit formats
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,

    // Normalized 16-bit per channel formats
    R16Unorm,
    R16Snorm,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Unorm,
    Rgba16Snorm,
}

/// Sub-class of the image type.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum ImageClass {
    /// Regular sampled image.
    Sampled {
        /// Kind of values to sample.
        kind: ScalarKind,
        /// Multi-sampled image.
        ///
        /// A multi-sampled image holds several samples per texel. Multi-sampled
        /// images cannot have mipmaps.
        multi: bool,
    },
    /// Depth comparison image.
    Depth {
        /// Multi-sampled depth image.
        multi: bool,
    },
    /// External texture.
    External,
    /// Storage image.
    Storage {
        format: StorageFormat,
        access: StorageAccess,
    },
}

/// The output topology for a mesh shader. Note that mesh shaders don't allow things like triangle-strips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum MeshOutputTopology {
    /// Outputs individual vertices to be rendered as points.
    Points,
    /// Outputs groups of 2 vertices to be renderedas lines .
    Lines,
    /// Outputs groups of 3 vertices to be rendered as triangles.
    Triangles,
}

/// Width of a boolean type, in bytes.
pub const BOOL_WIDTH: Bytes = 1;

/// Width of abstract types, in bytes.
pub const ABSTRACT_WIDTH: Bytes = 8;

impl ScalarKind {
    pub const fn is_numeric(self) -> bool {
        match self {
            ScalarKind::Sint
            | ScalarKind::Uint
            | ScalarKind::Float
            | ScalarKind::AbstractInt
            | ScalarKind::AbstractFloat => true,
            ScalarKind::Bool => false,
        }
    }
}

impl Scalar {
    pub const I16: Self = Self {
        kind: ScalarKind::Sint,
        width: 2,
    };
    pub const U16: Self = Self {
        kind: ScalarKind::Uint,
        width: 2,
    };
    pub const I32: Self = Self {
        kind: ScalarKind::Sint,
        width: 4,
    };
    pub const U32: Self = Self {
        kind: ScalarKind::Uint,
        width: 4,
    };
    pub const F16: Self = Self {
        kind: ScalarKind::Float,
        width: 2,
    };
    pub const F32: Self = Self {
        kind: ScalarKind::Float,
        width: 4,
    };
    pub const F64: Self = Self {
        kind: ScalarKind::Float,
        width: 8,
    };
    pub const I64: Self = Self {
        kind: ScalarKind::Sint,
        width: 8,
    };
    pub const U64: Self = Self {
        kind: ScalarKind::Uint,
        width: 8,
    };
    pub const BOOL: Self = Self {
        kind: ScalarKind::Bool,
        width: BOOL_WIDTH,
    };
    pub const ABSTRACT_INT: Self = Self {
        kind: ScalarKind::AbstractInt,
        width: ABSTRACT_WIDTH,
    };
    pub const ABSTRACT_FLOAT: Self = Self {
        kind: ScalarKind::AbstractFloat,
        width: ABSTRACT_WIDTH,
    };

    pub const fn is_abstract(self) -> bool {
        match self.kind {
            ScalarKind::AbstractInt | ScalarKind::AbstractFloat => true,
            ScalarKind::Sint | ScalarKind::Uint | ScalarKind::Float | ScalarKind::Bool => false,
        }
    }

    /// Construct a float `Scalar` with the given width.
    ///
    /// This is especially common when dealing with
    /// `TypeInner::Matrix`, where the scalar kind is implicit.
    pub const fn float(width: Bytes) -> Self {
        Self {
            kind: ScalarKind::Float,
            width,
        }
    }
}

impl From<StorageFormat> for Scalar {
    fn from(format: StorageFormat) -> Self {
        use {ScalarKind as Sk, StorageFormat as Sf};
        let kind = match format {
            Sf::R8Unorm => Sk::Float,
            Sf::R8Snorm => Sk::Float,
            Sf::R8Uint => Sk::Uint,
            Sf::R8Sint => Sk::Sint,
            Sf::R16Uint => Sk::Uint,
            Sf::R16Sint => Sk::Sint,
            Sf::R16Float => Sk::Float,
            Sf::Rg8Unorm => Sk::Float,
            Sf::Rg8Snorm => Sk::Float,
            Sf::Rg8Uint => Sk::Uint,
            Sf::Rg8Sint => Sk::Sint,
            Sf::R32Uint => Sk::Uint,
            Sf::R32Sint => Sk::Sint,
            Sf::R32Float => Sk::Float,
            Sf::Rg16Uint => Sk::Uint,
            Sf::Rg16Sint => Sk::Sint,
            Sf::Rg16Float => Sk::Float,
            Sf::Rgba8Unorm => Sk::Float,
            Sf::Rgba8Snorm => Sk::Float,
            Sf::Rgba8Uint => Sk::Uint,
            Sf::Rgba8Sint => Sk::Sint,
            Sf::Bgra8Unorm => Sk::Float,
            Sf::Rgb10a2Uint => Sk::Uint,
            Sf::Rgb10a2Unorm => Sk::Float,
            Sf::Rg11b10Ufloat => Sk::Float,
            Sf::R64Uint => Sk::Uint,
            Sf::Rg32Uint => Sk::Uint,
            Sf::Rg32Sint => Sk::Sint,
            Sf::Rg32Float => Sk::Float,
            Sf::Rgba16Uint => Sk::Uint,
            Sf::Rgba16Sint => Sk::Sint,
            Sf::Rgba16Float => Sk::Float,
            Sf::Rgba32Uint => Sk::Uint,
            Sf::Rgba32Sint => Sk::Sint,
            Sf::Rgba32Float => Sk::Float,
            Sf::R16Unorm => Sk::Float,
            Sf::R16Snorm => Sk::Float,
            Sf::Rg16Unorm => Sk::Float,
            Sf::Rg16Snorm => Sk::Float,
            Sf::Rgba16Unorm => Sk::Float,
            Sf::Rgba16Snorm => Sk::Float,
        };
        let width = match format {
            Sf::R64Uint => 8,
            _ => 4,
        };
        Scalar { kind, width }
    }
}

impl AddressSpace {
    pub fn access(self) -> StorageAccess {
        use StorageAccess as Sa;
        match self {
            AddressSpace::Function | AddressSpace::Private | AddressSpace::WorkGroup => {
                Sa::LOAD | Sa::STORE
            }
            AddressSpace::Uniform => Sa::LOAD,
            AddressSpace::Storage { access } => access,
            AddressSpace::Handle => Sa::LOAD,
            AddressSpace::Immediate => Sa::LOAD,
            // TaskPayload isn't always writable, but this is checked for elsewhere,
            // when not using multiple payloads and matching the entry payload is checked.
            AddressSpace::TaskPayload => Sa::LOAD | Sa::STORE,
            AddressSpace::RayPayload | AddressSpace::IncomingRayPayload => Sa::LOAD | Sa::STORE,
        }
    }
}

impl ImageClass {
    pub const fn is_multisampled(self) -> bool {
        match self {
            ImageClass::Sampled { multi, .. } | ImageClass::Depth { multi } => multi,
            ImageClass::Storage { .. } => false,
            ImageClass::External => false,
        }
    }

    pub const fn is_mipmapped(self) -> bool {
        match self {
            ImageClass::Sampled { multi, .. } | ImageClass::Depth { multi } => !multi,
            ImageClass::Storage { .. } => false,
            ImageClass::External => false,
        }
    }

    pub const fn is_depth(self) -> bool {
        matches!(self, ImageClass::Depth { .. })
    }
}

impl MeshOutputTopology {
    pub const fn to_builtin(self) -> BuiltIn {
        match self {
            Self::Points => BuiltIn::PointIndex,
            Self::Lines => BuiltIn::LineIndices,
            Self::Triangles => BuiltIn::TriangleIndices,
        }
    }
}

impl AddressSpace {
    pub const fn is_workgroup_like(self) -> bool {
        matches!(self, Self::WorkGroup | Self::TaskPayload)
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for StorageAccess {
    fn schema_name() -> alloc::borrow::Cow<'static, str> {
        "StorageAccess".into()
    }
    fn json_schema(g: &mut schemars::SchemaGenerator) -> schemars::Schema {
        <alloc::string::String as schemars::JsonSchema>::json_schema(g)
    }
}
