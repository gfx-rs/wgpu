/*!
Backend for [HLSL][hlsl] (High-Level Shading Language).

# Supported shader model versions:
- 5.0
- 5.1
- 6.0

# Layout of values in `uniform` buffers

WGSL's ["Internal Layout of Values"][ilov] rules specify how each WGSL
type should be stored in `uniform` and `storage` buffers. The HLSL we
generate must access values in that form, even when it is not what
HLSL would use normally.

The rules described here only apply to WGSL `uniform` variables. WGSL
`storage` buffers are translated as HLSL `ByteAddressBuffers`, for
which we generate `Load` and `Store` method calls with explicit byte
offsets. WGSL pipeline inputs must be scalars or vectors; they cannot
be matrices, which is where the interesting problems arise.

## Row- and column-major ordering for matrices

WGSL specifies that matrices in uniform buffers are stored in
column-major order. This matches HLSL's default, so one might expect
things to be straightforward. Unfortunately, WGSL and HLSL disagree on
what indexing a matrix means: in WGSL, `m[i]` retrieves the `i`'th
*column* of `m`, whereas in HLSL it retrieves the `i`'th *row*. We
want to avoid translating `m[i]` into some complicated reassembly of a
vector from individually fetched components, so this is a problem.

However, with a bit of trickery, it is possible to use HLSL's `m[i]`
as the translation of WGSL's `m[i]`:

- We declare all matrices in uniform buffers in HLSL with the
  `row_major` qualifier, and transpose the row and column counts: a
  WGSL `mat3x4<f32>`, say, becomes an HLSL `row_major float3x4`. (Note
  that WGSL and HLSL type names put the row and column in reverse
  order.) Since the HLSL type is the transpose of how WebGPU directs
  the user to store the data, HLSL will load all matrices transposed.

- Since matrices are transposed, an HLSL indexing expression retrieves
  the "columns" of the intended WGSL value, as desired.

- For vector-matrix multiplication, since `mul(transpose(m), v)` is
  equivalent to `mul(v, m)` (note the reversal of the arguments), and
  `mul(v, transpose(m))` is equivalent to `mul(m, v)`, we can
  translate WGSL `m * v` and `v * m` to HLSL by simply reversing the
  arguments to `mul`.

## Padding in two-row matrices

An HLSL `row_major floatKx2` matrix has padding between its rows that
the WGSL `matKx2<f32>` matrix it represents does not. HLSL stores all
matrix rows [aligned on 16-byte boundaries][16bb], whereas WGSL says
that the columns of a `matKx2<f32>` need only be [aligned as required
for `vec2<f32>`][ilov], which is [eight-byte alignment][8bb].

To compensate for this, any time a `matKx2<f32>` appears in a WGSL
`uniform` variable, whether directly as the variable's type or as part
of a struct/array, we actually emit `K` separate `float2` members, and
assemble/disassemble the matrix from its columns (in WGSL; rows in
HLSL) upon load and store.

For example, the following WGSL struct type:

```ignore
struct Baz {
        m: mat3x2<f32>,
}
```

is rendered as the HLSL struct type:

```ignore
struct Baz {
    float2 m_0; float2 m_1; float2 m_2;
};
```

The `wrapped_struct_matrix` functions in `help.rs` generate HLSL
helper functions to access such members, converting between the stored
form and the HLSL matrix types appropriately. For example, for reading
the member `m` of the `Baz` struct above, we emit:

```ignore
float3x2 GetMatmOnBaz(Baz obj) {
    return float3x2(obj.m_0, obj.m_1, obj.m_2);
}
```

We also emit an analogous `Set` function, as well as functions for
accessing individual columns by dynamic index.

[hlsl]: https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl
[ilov]: https://gpuweb.github.io/gpuweb/wgsl/#internal-value-layout
[16bb]: https://github.com/microsoft/DirectXShaderCompiler/wiki/Buffer-Packing#constant-buffer-packing
[8bb]: https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
*/

mod conv;
mod help;
mod keywords;
mod storage;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

use crate::{back, proc};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindTarget {
    pub space: u8,
    pub register: u32,
    /// If the binding is an unsized binding array, this overrides the size.
    pub binding_array_size: Option<u32>,
}

// Using `BTreeMap` instead of `HashMap` so that we can hash itself.
pub type BindingMap = std::collections::BTreeMap<crate::ResourceBinding, BindTarget>;

/// A HLSL shader model version.
#[allow(non_snake_case, non_camel_case_types)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum ShaderModel {
    V5_0,
    V5_1,
    V6_0,
}

impl ShaderModel {
    pub const fn to_str(self) -> &'static str {
        match self {
            Self::V5_0 => "5_0",
            Self::V5_1 => "5_1",
            Self::V6_0 => "6_0",
        }
    }
}

impl crate::ShaderStage {
    pub const fn to_hlsl_str(self) -> &'static str {
        match self {
            Self::Vertex => "vs",
            Self::Fragment => "ps",
            Self::Compute => "cs",
        }
    }
}

impl crate::ImageDimension {
    const fn to_hlsl_str(self) -> &'static str {
        match self {
            Self::D1 => "1D",
            Self::D2 => "2D",
            Self::D3 => "3D",
            Self::Cube => "Cube",
        }
    }
}

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum EntryPointError {
    #[error("mapping of {0:?} is missing")]
    MissingBinding(crate::ResourceBinding),
}

/// Configuration used in the [`Writer`].
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Options {
    /// The hlsl shader model to be used
    pub shader_model: ShaderModel,
    /// Map of resources association to binding locations.
    pub binding_map: BindingMap,
    /// Don't panic on missing bindings, instead generate any HLSL.
    pub fake_missing_bindings: bool,
    /// Add special constants to `SV_VertexIndex` and `SV_InstanceIndex`,
    /// to make them work like in Vulkan/Metal, with help of the host.
    pub special_constants_binding: Option<BindTarget>,
    /// Bind target of the push constant buffer
    pub push_constants_target: Option<BindTarget>,
    /// Should workgroup variables be zero initialized (by polyfilling)?
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_model: ShaderModel::V5_1,
            binding_map: BindingMap::default(),
            fake_missing_bindings: true,
            special_constants_binding: None,
            push_constants_target: None,
            zero_initialize_workgroup_memory: true,
        }
    }
}

impl Options {
    fn resolve_resource_binding(
        &self,
        res_binding: &crate::ResourceBinding,
    ) -> Result<BindTarget, EntryPointError> {
        match self.binding_map.get(res_binding) {
            Some(target) => Ok(target.clone()),
            None if self.fake_missing_bindings => Ok(BindTarget {
                space: res_binding.group as u8,
                register: res_binding.binding,
                binding_array_size: None,
            }),
            None => Err(EntryPointError::MissingBinding(res_binding.clone())),
        }
    }
}

/// Reflection info for entry point names.
#[derive(Default)]
pub struct ReflectionInfo {
    /// Mapping of the entry point names.
    ///
    /// Each item in the array corresponds to an entry point index. The real entry point name may be different if one of the
    /// reserved words are used.
    ///
    /// Note: Some entry points may fail translation because of missing bindings.
    pub entry_point_names: Vec<Result<String, EntryPointError>>,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IoError(#[from] FmtError),
    #[error("A scalar with an unsupported width was requested: {0:?}")]
    UnsupportedScalar(crate::Scalar),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
    #[error("{0}")]
    Custom(String),
}

#[derive(Default)]
struct Wrapped {
    array_lengths: crate::FastHashSet<help::WrappedArrayLength>,
    image_queries: crate::FastHashSet<help::WrappedImageQuery>,
    constructors: crate::FastHashSet<help::WrappedConstructor>,
    struct_matrix_access: crate::FastHashSet<help::WrappedStructMatrixAccess>,
    mat_cx2s: crate::FastHashSet<help::WrappedMatCx2>,
}

impl Wrapped {
    fn clear(&mut self) {
        self.array_lengths.clear();
        self.image_queries.clear();
        self.constructors.clear();
        self.struct_matrix_access.clear();
        self.mat_cx2s.clear();
    }
}

pub struct Writer<'a, W> {
    out: W,
    names: crate::FastHashMap<proc::NameKey, String>,
    namer: proc::Namer,
    /// HLSL backend options
    options: &'a Options,
    /// Information about entry point arguments and result types.
    entry_point_io: Vec<writer::EntryPointInterface>,
    /// Set of expressions that have associated temporary variables
    named_expressions: crate::NamedExpressions,
    wrapped: Wrapped,

    /// A reference to some part of a global variable, lowered to a series of
    /// byte offset calculations.
    ///
    /// See the [`storage`] module for background on why we need this.
    ///
    /// Each [`SubAccess`] in the vector is a lowering of some [`Access`] or
    /// [`AccessIndex`] expression to the level of byte strides and offsets. See
    /// [`SubAccess`] for details.
    ///
    /// This field is a member of [`Writer`] solely to allow re-use of
    /// the `Vec`'s dynamic allocation. The value is no longer needed
    /// once HLSL for the access has been generated.
    ///
    /// [`Storage`]: crate::AddressSpace::Storage
    /// [`SubAccess`]: storage::SubAccess
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    temp_access_chain: Vec<storage::SubAccess>,
    need_bake_expressions: back::NeedBakeExpressions,
}
