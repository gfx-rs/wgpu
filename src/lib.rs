/*! Universal shader translator.

The central structure of the crate is [`Module`]. A `Module` contains:

- [`Function`]s, which have arguments, a return type, local variables, and a body,

- [`EntryPoint`]s, which are specialized functions that can serve as the entry
  point for pipeline stages like vertex shading or fragment shading,

- [`Constant`]s and [`GlobalVariable`]s used by `EntryPoint`s and `Function`s, and

- [`Type`]s used by the above.

The body of an `EntryPoint` or `Function` is represented using two types:

- An [`Expression`] produces a value, but has no side effects or control flow.
  `Expressions` include variable references, unary and binary operators, and so
  on.

- A [`Statement`] can have side effects and structured control flow.
  `Statement`s do not produce a value, other than by storing one in some
  designated place. `Statements` include blocks, conditionals, and loops, but also
  operations that have side effects, like stores and function calls.

`Statement`s form a tree, with pointers into the DAG of `Expression`s.

Restricting side effects to statements simplifies analysis and code generation.
A Naga backend can generate code to evaluate an `Expression` however and
whenever it pleases, as long as it is certain to observe the side effects of all
previously executed `Statement`s.

Many `Statement` variants use the [`Block`] type, which is `Vec<Statement>`,
with optional span info, representing a series of statements executed in order. The body of an
`EntryPoint`s or `Function` is a `Block`, and `Statement` has a
[`Block`][Statement::Block] variant.

If the `clone` feature is enabled, [`Arena`], [`UniqueArena`], [`Type`], [`TypeInner`],
[`Constant`], [`Function`], [`EntryPoint`] and [`Module`] can be cloned.

## Arenas

To improve translator performance and reduce memory usage, most structures are
stored in an [`Arena`]. An `Arena<T>` stores a series of `T` values, indexed by
[`Handle<T>`](Handle) values, which are just wrappers around integer indexes.
For example, a `Function`'s expressions are stored in an `Arena<Expression>`,
and compound expressions refer to their sub-expressions via `Handle<Expression>`
values. (When examining the serialized form of a `Module`, note that the first
element of an `Arena` has an index of 1, not 0.)

A [`UniqueArena`] is just like an `Arena`, except that it stores only a single
instance of each value. The value type must implement `Eq` and `Hash`. Like an
`Arena`, inserting a value into a `UniqueArena` returns a `Handle` which can be
used to efficiently access the value, without a hash lookup. Inserting a value
multiple times returns the same `Handle`.

If the `span` feature is enabled, both `Arena` and `UniqueArena` can associate a
source code span with each element.

## Function Calls

Naga's representation of function calls is unusual. Most languages treat
function calls as expressions, but because calls may have side effects, Naga
represents them as a kind of statement, [`Statement::Call`]. If the function
returns a value, a call statement designates a particular [`Expression::CallResult`]
expression to represent its return value, for use by subsequent statements and
expressions.

## `Expression` evaluation time

It is essential to know when an [`Expression`] should be evaluated, because its
value may depend on previous [`Statement`]s' effects. But whereas the order of
execution for a tree of `Statement`s is apparent from its structure, it is not
so clear for `Expressions`, since an expression may be referred to by any number
of `Statement`s and other `Expression`s.

Naga's rules for when `Expression`s are evaluated are as follows:

-   [`Constant`](Expression::Constant) expressions are considered to be
    implicitly evaluated before execution begins.

-   [`FunctionArgument`] and [`LocalVariable`] expressions are considered
    implicitly evaluated upon entry to the function to which they belong.
    Function arguments cannot be assigned to, and `LocalVariable` expressions
    produce a *pointer to* the variable's value (for use with [`Load`] and
    [`Store`]). Neither varies while the function executes, so it suffices to
    consider these expressions evaluated once on entry.

-   Similarly, [`GlobalVariable`] expressions are considered implicitly
    evaluated before execution begins, since their value does not change while
    code executes, for one of two reasons:

    -   Most `GlobalVariable` expressions produce a pointer to the variable's
        value, for use with [`Load`] and [`Store`], as `LocalVariable`
        expressions do. Although the variable's value may change, its address
        does not.

    -   A `GlobalVariable` expression referring to a global in the
        [`AddressSpace::Handle`] address space produces the value directly, not
        a pointer. Such global variables hold opaque types like shaders or
        images, and cannot be assigned to.

-   A [`CallResult`] expression that is the `result` of a [`Statement::Call`],
    representing the call's return value, is evaluated when the `Call` statement
    is executed.

-   Similarly, an [`AtomicResult`] expression that is the `result` of an
    [`Atomic`] statement, representing the result of the atomic operation, is
    evaluated when the `Atomic` statement is executed.

-   All other expressions are evaluated when the (unique) [`Statement::Emit`]
    statement that covers them is executed. The [`Expression::needs_pre_emit`]
    method returns `true` if the given expression is one of those variants that
    does *not* need to be covered by an `Emit` statement.

Now, strictly speaking, not all `Expression` variants actually care when they're
evaluated. For example, you can evaluate a [`BinaryOperator::Add`] expression
any time you like, as long as you give it the right operands. It's really only a
very small set of expressions that are affected by timing:

-   [`Load`], [`ImageSample`], and [`ImageLoad`] expressions are influenced by
    stores to the variables or images they access, and must execute at the
    proper time relative to them.

-   [`Derivative`] expressions are sensitive to control flow uniformity: they
    must not be moved out of an area of uniform control flow into a non-uniform
    area.

-   More generally, any expression that's used by more than one other expression
    or statement should probably be evaluated only once, and then stored in a
    variable to be cited at each point of use.

Naga tries to help back ends handle all these cases correctly in a somewhat
circuitous way. The [`ModuleInfo`] structure returned by [`Validator::validate`]
provides a reference count for each expression in each function in the module.
Naturally, any expression with a reference count of two or more deserves to be
evaluated and stored in a temporary variable at the point that the `Emit`
statement covering it is executed. But if we selectively lower the reference
count threshold to _one_ for the sensitive expression types listed above, so
that we _always_ generate a temporary variable and save their value, then the
same code that manages multiply referenced expressions will take care of
introducing temporaries for time-sensitive expressions as well. The
`Expression::bake_ref_count` method (private to the back ends) is meant to help
with this.

## `Expression` scope

Each `Expression` has a *scope*, which is the region of the function within
which it can be used by `Statement`s and other `Expression`s. It is a validation
error to use an `Expression` outside its scope.

An expression's scope is defined as follows:

-   The scope of a [`Constant`], [`GlobalVariable`], [`FunctionArgument`] or
    [`LocalVariable`] expression covers the entire `Function` in which it
    occurs.

-   The scope of an expression evaluated by an [`Emit`] statement covers the
    subsequent expressions in that `Emit`, the subsequent statements in the `Block`
    to which that `Emit` belongs (if any) and their sub-statements (if any).

-   The `result` expression of a [`Call`] or [`Atomic`] statement has a scope
    covering the subsequent statements in the `Block` in which the statement
    occurs (if any) and their sub-statements (if any).

For example, this implies that an expression evaluated by some statement in a
nested `Block` is not available in the `Block`'s parents. Such a value would
need to be stored in a local variable to be carried upwards in the statement
tree.

[`AtomicResult`]: Expression::AtomicResult
[`CallResult`]: Expression::CallResult
[`Constant`]: Expression::Constant
[`Derivative`]: Expression::Derivative
[`FunctionArgument`]: Expression::FunctionArgument
[`GlobalVariable`]: Expression::GlobalVariable
[`ImageLoad`]: Expression::ImageLoad
[`ImageSample`]: Expression::ImageSample
[`Load`]: Expression::Load
[`LocalVariable`]: Expression::LocalVariable

[`Atomic`]: Statement::Atomic
[`Call`]: Statement::Call
[`Emit`]: Statement::Emit
[`Store`]: Statement::Store

[`Validator::validate`]: valid::Validator::validate
[`ModuleInfo`]: valid::ModuleInfo
*/

#![allow(
    clippy::new_without_default,
    clippy::unneeded_field_pattern,
    clippy::match_like_matches_macro,
    clippy::if_same_then_else,
    clippy::derive_partial_eq_without_eq,
    clippy::only_used_in_recursion,
    clippy::needless_borrowed_reference,
    clippy::useless_conversion,
    clippy::needless_lifetimes,
    clippy::bool_to_int_with_if
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    clippy::pattern_type_mismatch,
    clippy::missing_const_for_fn
)]
#![deny(clippy::panic)]

mod arena;
pub mod back;
mod block;
pub mod front;
pub mod keywords;
pub mod proc;
mod span;
pub mod valid;

pub use crate::arena::{Arena, Handle, Range, UniqueArena};

pub use crate::span::{SourceLocation, Span, SpanContext, WithSpan};
#[cfg(feature = "arbitrary")]
use arbitrary::Arbitrary;
#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

/// Width of a boolean type, in bytes.
pub const BOOL_WIDTH: Bytes = 1;

/// Hash map that is faster but not resilient to DoS attacks.
pub type FastHashMap<K, T> = rustc_hash::FxHashMap<K, T>;
/// Hash set that is faster but not resilient to DoS attacks.
pub type FastHashSet<K> = rustc_hash::FxHashSet<K>;

/// Insertion-order-preserving hash set (`IndexSet<K>`), but with the same
/// hasher as `FastHashSet<K>` (faster but not resilient to DoS attacks).
pub type FastIndexSet<K> =
    indexmap::IndexSet<K, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Map of expressions that have associated variable names
pub(crate) type NamedExpressions = FastHashMap<Handle<Expression>, String>;

/// Early fragment tests.
///
/// In a standard situation, if a driver determines that it is possible to switch on early depth test, it will.
///
/// Typical situations when early depth test is switched off:
///   - Calling `discard` in a shader.
///   - Writing to the depth buffer, unless ConservativeDepth is enabled.
///
/// To use in a shader:
///   - GLSL: `layout(early_fragment_tests) in;`
///   - HLSL: `Attribute earlydepthstencil`
///   - SPIR-V: `ExecutionMode EarlyFragmentTests`
///
/// For more, see:
///   - <https://www.khronos.org/opengl/wiki/Early_Fragment_Test#Explicit_specification>
///   - <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-earlydepthstencil>
///   - <https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#Execution_Mode>
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct EarlyDepthTest {
    conservative: Option<ConservativeDepth>,
}
/// Enables adjusting depth without disabling early Z.
///
/// To use in a shader:
///   - GLSL: `layout (depth_<greater/less/unchanged/any>) out float gl_FragDepth;`
///     - `depth_any` option behaves as if the layout qualifier was not present.
///   - HLSL: `SV_DepthGreaterEqual`/`SV_DepthLessEqual`/`SV_Depth`
///   - SPIR-V: `ExecutionMode Depth<Greater/Less/Unchanged>`
///
/// For more, see:
///   - <https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_conservative_depth.txt>
///   - <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-semantics#system-value-semantics>
///   - <https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#Execution_Mode>
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ConservativeDepth {
    /// Shader may rewrite depth only with a value greater than calculated.
    GreaterEqual,

    /// Shader may rewrite depth smaller than one that would have been written without the modification.
    LessEqual,

    /// Shader may not rewrite depth value.
    Unchanged,
}

/// Stage of the programmable pipeline.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[allow(missing_docs)] // The names are self evident
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// Addressing space of variables.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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
    /// Push constants.
    PushConstant,
}

/// Built-in inputs and outputs.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum BuiltIn {
    Position { invariant: bool },
    ViewIndex,
    // vertex
    BaseInstance,
    BaseVertex,
    ClipDistance,
    CullDistance,
    InstanceIndex,
    PointSize,
    VertexIndex,
    // fragment
    FragDepth,
    FrontFacing,
    PrimitiveIndex,
    SampleIndex,
    SampleMask,
    // compute
    GlobalInvocationId,
    LocalInvocationId,
    LocalInvocationIndex,
    WorkGroupId,
    WorkGroupSize,
    NumWorkGroups,
}

/// Number of bytes per scalar.
pub type Bytes = u8;

/// Number of components in a vector.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum VectorSize {
    /// 2D vector
    Bi = 2,
    /// 3D vector
    Tri = 3,
    /// 4D vector
    Quad = 4,
}

/// Primitive type for a scalar.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ScalarKind {
    /// Signed integer type.
    Sint,
    /// Unsigned integer type.
    Uint,
    /// Floating point type.
    Float,
    /// Boolean type.
    Bool,
}

/// Size of an array.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ArraySize {
    /// The array size is constant.
    Constant(Handle<Constant>),
    /// The array size can change at runtime.
    Dynamic,
}

/// The interpolation qualifier of a binding or struct field.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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
}

/// The sampling qualifiers of a binding or struct field.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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
}

/// Member of a user-defined structure.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct StructMember {
    pub name: Option<String>,
    /// Type of the field.
    pub ty: Handle<Type>,
    /// For I/O structs, defines the binding.
    pub binding: Option<Binding>,
    /// Offset from the beginning from the struct.
    pub offset: u32,
}

/// The number of dimensions an image has.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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
    #[cfg_attr(feature = "serialize", derive(Serialize))]
    #[cfg_attr(feature = "deserialize", derive(Deserialize))]
    #[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
    #[derive(Default)]
    pub struct StorageAccess: u32 {
        /// Storage can be used as a source for load ops.
        const LOAD = 0x1;
        /// Storage can be used as a target for store ops.
        const STORE = 0x2;
    }
}

/// Image storage format.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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

    // Packed 32-bit formats
    Rgb10a2Unorm,
    Rg11b10Float,

    // 64-bit formats
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
}

/// Sub-class of the image type.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
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
    /// Storage image.
    Storage {
        format: StorageFormat,
        access: StorageAccess,
    },
}

/// A data type declared in the module.
#[derive(Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct Type {
    /// The name of the type, if any.
    pub name: Option<String>,
    /// Inner structure that depends on the kind of the type.
    pub inner: TypeInner,
}

/// Enum with additional information, depending on the kind of type.
#[derive(Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum TypeInner {
    /// Number of integral or floating-point kind.
    Scalar { kind: ScalarKind, width: Bytes },
    /// Vector of numbers.
    Vector {
        size: VectorSize,
        kind: ScalarKind,
        width: Bytes,
    },
    /// Matrix of floats.
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        width: Bytes,
    },
    /// Atomic scalar.
    Atomic { kind: ScalarKind, width: Bytes },
    /// Pointer to another type.
    ///
    /// Pointers to scalars and vectors should be treated as equivalent to
    /// [`ValuePointer`] types. Use the [`TypeInner::equivalent`] method to
    /// compare types in a way that treats pointers correctly.
    ///
    /// ## Pointers to non-`SIZED` types
    ///
    /// The `base` type of a pointer may be a non-[`SIZED`] type like a
    /// dynamically-sized [`Array`], or a [`Struct`] whose last member is a
    /// dynamically sized array. Such pointers occur as the types of
    /// [`GlobalVariable`] or [`AccessIndex`] expressions referring to
    /// dynamically-sized arrays.
    ///
    /// However, among pointers to non-`SIZED` types, only pointers to `Struct`s
    /// are [`DATA`]. Pointers to dynamically sized `Array`s cannot be passed as
    /// arguments, stored in variables, or held in arrays or structures. Their
    /// only use is as the types of `AccessIndex` expressions.
    ///
    /// [`SIZED`]: valid::TypeFlags::SIZED
    /// [`DATA`]: valid::TypeFlags::DATA
    /// [`Array`]: TypeInner::Array
    /// [`Struct`]: TypeInner::Struct
    /// [`ValuePointer`]: TypeInner::ValuePointer
    /// [`GlobalVariable`]: Expression::GlobalVariable
    /// [`AccessIndex`]: Expression::AccessIndex
    Pointer {
        base: Handle<Type>,
        space: AddressSpace,
    },

    /// Pointer to a scalar or vector.
    ///
    /// A `ValuePointer` type is equivalent to a `Pointer` whose `base` is a
    /// `Scalar` or `Vector` type. This is for use in [`TypeResolution::Value`]
    /// variants; see the documentation for [`TypeResolution`] for details.
    ///
    /// Use the [`TypeInner::equivalent`] method to compare types that could be
    /// pointers, to ensure that `Pointer` and `ValuePointer` types are
    /// recognized as equivalent.
    ///
    /// [`TypeResolution`]: proc::TypeResolution
    /// [`TypeResolution::Value`]: proc::TypeResolution::Value
    ValuePointer {
        size: Option<VectorSize>,
        kind: ScalarKind,
        width: Bytes,
        space: AddressSpace,
    },

    /// Homogenous list of elements.
    ///
    /// The `base` type must be a [`SIZED`], [`DATA`] type.
    ///
    /// ## Dynamically sized arrays
    ///
    /// An `Array` is [`SIZED`] unless its `size` is [`Dynamic`].
    /// Dynamically-sized arrays may only appear in a few situations:
    ///
    /// -   They may appear as the type of a [`GlobalVariable`], or as the last
    ///     member of a [`Struct`].
    ///
    /// -   They may appear as the base type of a [`Pointer`]. An
    ///     [`AccessIndex`] expression referring to a struct's final
    ///     unsized array member would have such a pointer type. However, such
    ///     pointer types may only appear as the types of such intermediate
    ///     expressions. They are not [`DATA`], and cannot be stored in
    ///     variables, held in arrays or structs, or passed as parameters.
    ///
    /// [`SIZED`]: crate::valid::TypeFlags::SIZED
    /// [`DATA`]: crate::valid::TypeFlags::DATA
    /// [`Dynamic`]: ArraySize::Dynamic
    /// [`Struct`]: TypeInner::Struct
    /// [`Pointer`]: TypeInner::Pointer
    /// [`AccessIndex`]: Expression::AccessIndex
    Array {
        base: Handle<Type>,
        size: ArraySize,
        stride: u32,
    },

    /// User-defined structure.
    ///
    /// There must always be at least one member.
    ///
    /// A `Struct` type is [`DATA`], and the types of its members must be
    /// `DATA` as well.
    ///
    /// Member types must be [`SIZED`], except for the final member of a
    /// struct, which may be a dynamically sized [`Array`]. The
    /// `Struct` type itself is `SIZED` when all its members are `SIZED`.
    ///
    /// [`DATA`]: crate::valid::TypeFlags::DATA
    /// [`SIZED`]: crate::valid::TypeFlags::SIZED
    /// [`Array`]: TypeInner::Array
    Struct {
        members: Vec<StructMember>,
        //TODO: should this be unaligned?
        span: u32,
    },
    /// Possibly multidimensional array of texels.
    Image {
        dim: ImageDimension,
        arrayed: bool,
        //TODO: consider moving `multisampled: bool` out
        class: ImageClass,
    },
    /// Can be used to sample values from images.
    Sampler { comparison: bool },

    /// Array of bindings.
    ///
    /// A `BindingArray` represents an array where each element draws its value
    /// from a separate bound resource. The array's element type `base` may be
    /// [`Image`], [`Sampler`], or any type that would be permitted for a global
    /// in the [`Uniform`] or [`Storage`] address spaces. Only global variables
    /// may be binding arrays; on the host side, their values are provided by
    /// [`TextureViewArray`], [`SamplerArray`], or [`BufferArray`]
    /// bindings.
    ///
    /// Since each element comes from a distinct resource, a binding array of
    /// images could have images of varying sizes (but not varying dimensions;
    /// they must all have the same `Image` type). Or, a binding array of
    /// buffers could have elements that are dynamically sized arrays, each with
    /// a different length.
    ///
    /// Binding arrays are not [`DATA`]. This means that all binding array
    /// globals must be placed in the [`Handle`] address space. Referring to
    /// such a global produces a `BindingArray` value directly; there are never
    /// pointers to binding arrays. The only operation permitted on
    /// `BindingArray` values is indexing, which yields the element by value,
    /// not a pointer to the element. (This means that buffer array contents
    /// cannot be stored to; [naga#1864] covers lifting this restriction.)
    ///
    /// Unlike textures and samplers, binding arrays are not [`ARGUMENT`], so
    /// they cannot be passed as arguments to functions.
    ///
    /// Naga's WGSL front end supports binding arrays with the type syntax
    /// `binding_array<T, N>`.
    ///
    /// [`Image`]: TypeInner::Image
    /// [`Sampler`]: TypeInner::Sampler
    /// [`Uniform`]: AddressSpace::Uniform
    /// [`Storage`]: AddressSpace::Storage
    /// [`TextureViewArray`]: https://docs.rs/wgpu/latest/wgpu/enum.BindingResource.html#variant.TextureViewArray
    /// [`SamplerArray`]: https://docs.rs/wgpu/latest/wgpu/enum.BindingResource.html#variant.SamplerArray
    /// [`BufferArray`]: https://docs.rs/wgpu/latest/wgpu/enum.BindingResource.html#variant.BufferArray
    /// [`DATA`]: crate::valid::TypeFlags::DATA
    /// [`Handle`]: AddressSpace::Handle
    /// [`ARGUMENT`]: crate::valid::TypeFlags::ARGUMENT
    /// [naga#1864]: https://github.com/gfx-rs/naga/issues/1864
    BindingArray { base: Handle<Type>, size: ArraySize },
}

/// Constant value.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct Constant {
    pub name: Option<String>,
    pub specialization: Option<u32>,
    pub inner: ConstantInner,
}

/// A literal scalar value, used in constants.
#[derive(Debug, Clone, Copy, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ScalarValue {
    Sint(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
}

/// Additional information, dependent on the kind of constant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ConstantInner {
    Scalar {
        width: Bytes,
        value: ScalarValue,
    },
    Composite {
        ty: Handle<Type>,
        components: Vec<Handle<Constant>>,
    },
}

/// Describes how an input/output variable is to be bound.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum Binding {
    /// Built-in shader variable.
    BuiltIn(BuiltIn),

    /// Indexed location.
    ///
    /// Values passed from the [`Vertex`] stage to the [`Fragment`] stage must
    /// have their `interpolation` defaulted (i.e. not `None`) by the front end
    /// as appropriate for that language.
    ///
    /// For other stages, we permit interpolations even though they're ignored.
    /// When a front end is parsing a struct type, it usually doesn't know what
    /// stages will be using it for IO, so it's easiest if it can apply the
    /// defaults to anything with a `Location` binding, just in case.
    ///
    /// For anything other than floating-point scalars and vectors, the
    /// interpolation must be `Flat`.
    ///
    /// [`Vertex`]: crate::ShaderStage::Vertex
    /// [`Fragment`]: crate::ShaderStage::Fragment
    Location {
        location: u32,
        interpolation: Option<Interpolation>,
        sampling: Option<Sampling>,
    },
}

/// Pipeline binding information for global resources.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct ResourceBinding {
    /// The bind group index.
    pub group: u32,
    /// Binding number within the group.
    pub binding: u32,
}

/// Variable defined at module level.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct GlobalVariable {
    /// Name of the variable, if any.
    pub name: Option<String>,
    /// How this variable is to be stored.
    pub space: AddressSpace,
    /// For resources, defines the binding point.
    pub binding: Option<ResourceBinding>,
    /// The type of this variable.
    pub ty: Handle<Type>,
    /// Initial value for this variable.
    pub init: Option<Handle<Constant>>,
}

/// Variable defined at function level.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct LocalVariable {
    /// Name of the variable, if any.
    pub name: Option<String>,
    /// The type of this variable.
    pub ty: Handle<Type>,
    /// Initial value for this variable.
    pub init: Option<Handle<Constant>>,
}

/// Operation that can be applied on a single value.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum UnaryOperator {
    Negate,
    Not,
}

/// Operation that can be applied on two values.
///
/// ## Arithmetic type rules
///
/// The arithmetic operations `Add`, `Subtract`, `Multiply`, `Divide`, and
/// `Modulo` can all be applied to [`Scalar`] types other than [`Bool`], or
/// [`Vector`]s thereof. Both operands must have the same type.
///
/// `Add` and `Subtract` can also be applied to [`Matrix`] values. Both operands
/// must have the same type.
///
/// `Multiply` supports additional cases:
///
/// -   A [`Matrix`] or [`Vector`] can be multiplied by a scalar [`Float`],
///     either on the left or the right.
///
/// -   A [`Matrix`] on the left can be multiplied by a [`Vector`] on the right
///     if the matrix has as many columns as the vector has components (`matCxR
///     * VecC`).
///
/// -   A [`Vector`] on the left can be multiplied by a [`Matrix`] on the right
///     if the matrix has as many rows as the vector has components (`VecR *
///     matCxR`).
///
/// -   Two matrices can be multiplied if the left operand has as many columns
///     as the right operand has rows (`matNxR * matCxN`).
///
/// In all the above `Multiply` cases, the byte widths of the underlying scalar
/// types of both operands must be the same.
///
/// Note that `Multiply` supports mixed vector and scalar operations directly,
/// whereas the other arithmetic operations require an explicit [`Splat`] for
/// mixed-type use.
///
/// [`Scalar`]: TypeInner::Scalar
/// [`Vector`]: TypeInner::Vector
/// [`Matrix`]: TypeInner::Matrix
/// [`Float`]: ScalarKind::Float
/// [`Bool`]: ScalarKind::Bool
/// [`Splat`]: Expression::Splat
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    /// Equivalent of the WGSL's `%` operator or SPIR-V's `OpFRem`
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    ExclusiveOr,
    InclusiveOr,
    LogicalAnd,
    LogicalOr,
    ShiftLeft,
    /// Right shift carries the sign of signed integers only.
    ShiftRight,
}

/// Function on an atomic value.
///
/// Note: these do not include load/store, which use the existing
/// [`Expression::Load`] and [`Statement::Store`].
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum AtomicFunction {
    Add,
    Subtract,
    And,
    ExclusiveOr,
    InclusiveOr,
    Min,
    Max,
    Exchange { compare: Option<Handle<Expression>> },
}

/// Axis on which to compute a derivative.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum DerivativeAxis {
    X,
    Y,
    Width,
}

/// Built-in shader function for testing relation between values.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum RelationalFunction {
    All,
    Any,
    IsNan,
    IsInf,
    IsFinite,
    IsNormal,
}

/// Built-in shader function for math.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum MathFunction {
    // comparison
    Abs,
    Min,
    Max,
    Clamp,
    Saturate,
    // trigonometry
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,
    Acos,
    Asin,
    Atan,
    Atan2,
    Asinh,
    Acosh,
    Atanh,
    Radians,
    Degrees,
    // decomposition
    Ceil,
    Floor,
    Round,
    Fract,
    Trunc,
    Modf,
    Frexp,
    Ldexp,
    // exponent
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
    // geometry
    Dot,
    Outer,
    Cross,
    Distance,
    Length,
    Normalize,
    FaceForward,
    Reflect,
    Refract,
    // computational
    Sign,
    Fma,
    Mix,
    Step,
    SmoothStep,
    Sqrt,
    InverseSqrt,
    Inverse,
    Transpose,
    Determinant,
    // bits
    CountOneBits,
    ReverseBits,
    ExtractBits,
    InsertBits,
    FindLsb,
    FindMsb,
    // data packing
    Pack4x8snorm,
    Pack4x8unorm,
    Pack2x16snorm,
    Pack2x16unorm,
    Pack2x16float,
    // data unpacking
    Unpack4x8snorm,
    Unpack4x8unorm,
    Unpack2x16snorm,
    Unpack2x16unorm,
    Unpack2x16float,
}

/// Sampling modifier to control the level of detail.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum SampleLevel {
    Auto,
    Zero,
    Exact(Handle<Expression>),
    Bias(Handle<Expression>),
    Gradient {
        x: Handle<Expression>,
        y: Handle<Expression>,
    },
}

/// Type of an image query.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum ImageQuery {
    /// Get the size at the specified level.
    Size {
        /// If `None`, the base level is considered.
        level: Option<Handle<Expression>>,
    },
    /// Get the number of mipmap levels.
    NumLevels,
    /// Get the number of array layers.
    NumLayers,
    /// Get the number of samples.
    NumSamples,
}

/// Component selection for a vector swizzle.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum SwizzleComponent {
    ///
    X = 0,
    ///
    Y = 1,
    ///
    Z = 2,
    ///
    W = 3,
}

bitflags::bitflags! {
    /// Memory barrier flags.
    #[cfg_attr(feature = "serialize", derive(Serialize))]
    #[cfg_attr(feature = "deserialize", derive(Deserialize))]
    #[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
    #[derive(Default)]
    pub struct Barrier: u32 {
        /// Barrier affects all `AddressSpace::Storage` accesses.
        const STORAGE = 0x1;
        /// Barrier affects all `AddressSpace::WorkGroup` accesses.
        const WORK_GROUP = 0x2;
    }
}

/// An expression that can be evaluated to obtain a value.
///
/// This is a Single Static Assignment (SSA) scheme similar to SPIR-V.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum Expression {
    /// Array access with a computed index.
    ///
    /// ## Typing rules
    ///
    /// The `base` operand must be some composite type: [`Vector`], [`Matrix`],
    /// [`Array`], a [`Pointer`] to one of those, or a [`ValuePointer`] with a
    /// `size`.
    ///
    /// The `index` operand must be an integer, signed or unsigned.
    ///
    /// Indexing a [`Vector`] or [`Array`] produces a value of its element type.
    /// Indexing a [`Matrix`] produces a [`Vector`].
    ///
    /// Indexing a [`Pointer`] to any of the above produces a pointer to the
    /// element/component type, in the same [`space`]. In the case of [`Array`],
    /// the result is an actual [`Pointer`], but for vectors and matrices, there
    /// may not be any type in the arena representing the component's type, so
    /// those produce [`ValuePointer`] types equivalent to the appropriate
    /// [`Pointer`].
    ///
    /// ## Dynamic indexing restrictions
    ///
    /// To accommodate restrictions in some of the shader languages that Naga
    /// targets, it is not permitted to subscript a matrix or array with a
    /// dynamically computed index unless that matrix or array appears behind a
    /// pointer. In other words, if the inner type of `base` is [`Array`] or
    /// [`Matrix`], then `index` must be a constant. But if the type of `base`
    /// is a [`Pointer`] to an array or matrix or a [`ValuePointer`] with a
    /// `size`, then the index may be any expression of integer type.
    ///
    /// You can use the [`Expression::is_dynamic_index`] method to determine
    /// whether a given index expression requires matrix or array base operands
    /// to be behind a pointer.
    ///
    /// (It would be simpler to always require the use of `AccessIndex` when
    /// subscripting arrays and matrices that are not behind pointers, but to
    /// accommodate existing front ends, Naga also permits `Access`, with a
    /// restricted `index`.)
    ///
    /// [`Vector`]: TypeInner::Vector
    /// [`Matrix`]: TypeInner::Matrix
    /// [`Array`]: TypeInner::Array
    /// [`Pointer`]: TypeInner::Pointer
    /// [`space`]: TypeInner::Pointer::space
    /// [`ValuePointer`]: TypeInner::ValuePointer
    /// [`Float`]: ScalarKind::Float
    Access {
        base: Handle<Expression>,
        index: Handle<Expression>,
    },
    /// Access the same types as [`Access`], plus [`Struct`] with a known index.
    ///
    /// [`Access`]: Expression::Access
    /// [`Struct`]: TypeInner::Struct
    AccessIndex {
        base: Handle<Expression>,
        index: u32,
    },
    /// Constant value.
    Constant(Handle<Constant>),
    /// Splat scalar into a vector.
    Splat {
        size: VectorSize,
        value: Handle<Expression>,
    },
    /// Vector swizzle.
    Swizzle {
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: [SwizzleComponent; 4],
    },
    /// Composite expression.
    Compose {
        ty: Handle<Type>,
        components: Vec<Handle<Expression>>,
    },

    /// Reference a function parameter, by its index.
    ///
    /// A `FunctionArgument` expression evaluates to a pointer to the argument's
    /// value. You must use a [`Load`] expression to retrieve its value, or a
    /// [`Store`] statement to assign it a new value.
    ///
    /// [`Load`]: Expression::Load
    /// [`Store`]: Statement::Store
    FunctionArgument(u32),

    /// Reference a global variable.
    ///
    /// If the given `GlobalVariable`'s [`space`] is [`AddressSpace::Handle`],
    /// then the variable stores some opaque type like a sampler or an image,
    /// and a `GlobalVariable` expression referring to it produces the
    /// variable's value directly.
    ///
    /// For any other address space, a `GlobalVariable` expression produces a
    /// pointer to the variable's value. You must use a [`Load`] expression to
    /// retrieve its value, or a [`Store`] statement to assign it a new value.
    ///
    /// [`space`]: GlobalVariable::space
    /// [`Load`]: Expression::Load
    /// [`Store`]: Statement::Store
    GlobalVariable(Handle<GlobalVariable>),

    /// Reference a local variable.
    ///
    /// A `LocalVariable` expression evaluates to a pointer to the variable's value.
    /// You must use a [`Load`](Expression::Load) expression to retrieve its value,
    /// or a [`Store`](Statement::Store) statement to assign it a new value.
    LocalVariable(Handle<LocalVariable>),

    /// Load a value indirectly.
    ///
    /// For [`TypeInner::Atomic`] the result is a corresponding scalar.
    /// For other types behind the `pointer<T>`, the result is T.
    Load { pointer: Handle<Expression> },
    /// Sample a point from a sampled or a depth image.
    ImageSample {
        image: Handle<Expression>,
        sampler: Handle<Expression>,
        /// If Some(), this operation is a gather operation
        /// on the selected component.
        gather: Option<SwizzleComponent>,
        coordinate: Handle<Expression>,
        array_index: Option<Handle<Expression>>,
        offset: Option<Handle<Constant>>,
        level: SampleLevel,
        depth_ref: Option<Handle<Expression>>,
    },

    /// Load a texel from an image.
    ///
    /// For most images, this returns a four-element vector of the same
    /// [`ScalarKind`] as the image. If the format of the image does not have
    /// four components, default values are provided: the first three components
    /// (typically R, G, and B) default to zero, and the final component
    /// (typically alpha) defaults to one.
    ///
    /// However, if the image's [`class`] is [`Depth`], then this returns a
    /// [`Float`] scalar value.
    ///
    /// [`ScalarKind`]: ScalarKind
    /// [`class`]: TypeInner::Image::class
    /// [`Depth`]: ImageClass::Depth
    /// [`Float`]: ScalarKind::Float
    ImageLoad {
        /// The image to load a texel from. This must have type [`Image`]. (This
        /// will necessarily be a [`GlobalVariable`] or [`FunctionArgument`]
        /// expression, since no other expressions are allowed to have that
        /// type.)
        ///
        /// [`Image`]: TypeInner::Image
        /// [`GlobalVariable`]: Expression::GlobalVariable
        /// [`FunctionArgument`]: Expression::FunctionArgument
        image: Handle<Expression>,

        /// The coordinate of the texel we wish to load. This must be a scalar
        /// for [`D1`] images, a [`Bi`] vector for [`D2`] images, and a [`Tri`]
        /// vector for [`D3`] images. (Array indices, sample indices, and
        /// explicit level-of-detail values are supplied separately.) Its
        /// component type must be [`Sint`].
        ///
        /// [`D1`]: ImageDimension::D1
        /// [`D2`]: ImageDimension::D2
        /// [`D3`]: ImageDimension::D3
        /// [`Bi`]: VectorSize::Bi
        /// [`Tri`]: VectorSize::Tri
        /// [`Sint`]: ScalarKind::Sint
        coordinate: Handle<Expression>,

        /// The index into an arrayed image. If the [`arrayed`] flag in
        /// `image`'s type is `true`, then this must be `Some(expr)`, where
        /// `expr` is a [`Sint`] scalar. Otherwise, it must be `None`.
        ///
        /// [`arrayed`]: TypeInner::Image::arrayed
        /// [`Sint`]: ScalarKind::Sint
        array_index: Option<Handle<Expression>>,

        /// A sample index, for multisampled [`Sampled`] and [`Depth`] images.
        ///
        /// [`Sampled`]: ImageClass::Sampled
        /// [`Depth`]: ImageClass::Depth
        sample: Option<Handle<Expression>>,

        /// A level of detail, for mipmapped images.
        ///
        /// This must be present when accessing non-multisampled
        /// [`Sampled`] and [`Depth`] images, even if only the
        /// full-resolution level is present (in which case the only
        /// valid level is zero).
        ///
        /// [`Sampled`]: ImageClass::Sampled
        /// [`Depth`]: ImageClass::Depth
        level: Option<Handle<Expression>>,
    },

    /// Query information from an image.
    ImageQuery {
        image: Handle<Expression>,
        query: ImageQuery,
    },
    /// Apply an unary operator.
    Unary {
        op: UnaryOperator,
        expr: Handle<Expression>,
    },
    /// Apply a binary operator.
    Binary {
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    },
    /// Select between two values based on a condition.
    ///
    /// Note that, because expressions have no side effects, it is unobservable
    /// whether the non-selected branch is evaluated.
    Select {
        /// Boolean expression
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    },
    /// Compute the derivative on an axis.
    Derivative {
        axis: DerivativeAxis,
        //modifier,
        expr: Handle<Expression>,
    },
    /// Call a relational function.
    Relational {
        fun: RelationalFunction,
        argument: Handle<Expression>,
    },
    /// Call a math function
    Math {
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        arg3: Option<Handle<Expression>>,
    },
    /// Cast a simple type to another kind.
    As {
        /// Source expression, which can only be a scalar or a vector.
        expr: Handle<Expression>,
        /// Target scalar kind.
        kind: ScalarKind,
        /// If provided, converts to the specified byte width.
        /// Otherwise, bitcast.
        convert: Option<Bytes>,
    },
    /// Result of calling another function.
    CallResult(Handle<Function>),
    /// Result of an atomic operation.
    AtomicResult {
        kind: ScalarKind,
        width: Bytes,
        comparison: bool,
    },
    /// Get the length of an array.
    /// The expression must resolve to a pointer to an array with a dynamic size.
    ///
    /// This doesn't match the semantics of spirv's `OpArrayLength`, which must be passed
    /// a pointer to a structure containing a runtime array in its' last field.
    ArrayLength(Handle<Expression>),
}

pub use block::Block;

/// The value of the switch case.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum SwitchValue {
    Integer(i32),
    Default,
}

/// A case for a switch statement.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct SwitchCase {
    /// Value, upon which the case is considered true.
    pub value: SwitchValue,
    /// Body of the case.
    pub body: Block,
    /// If true, the control flow continues to the next case in the list,
    /// or default.
    pub fall_through: bool,
}

//TODO: consider removing `Clone`. It's not valid to clone `Statement::Emit` anyway.
/// Instructions which make up an executable block.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub enum Statement {
    /// Emit a range of expressions, visible to all statements that follow in this block.
    ///
    /// See the [module-level documentation][emit] for details.
    ///
    /// [emit]: index.html#expression-evaluation-time
    Emit(Range<Expression>),
    /// A block containing more statements, to be executed sequentially.
    Block(Block),
    /// Conditionally executes one of two blocks, based on the value of the condition.
    If {
        condition: Handle<Expression>, //bool
        accept: Block,
        reject: Block,
    },
    /// Conditionally executes one of multiple blocks, based on the value of the selector.
    Switch {
        selector: Handle<Expression>, //int
        cases: Vec<SwitchCase>,
    },

    /// Executes a block repeatedly.
    ///
    /// Each iteration of the loop executes the `body` block, followed by the
    /// `continuing` block.
    ///
    /// Executing a [`Break`], [`Return`] or [`Kill`] statement exits the loop.
    ///
    /// A [`Continue`] statement in `body` jumps to the `continuing` block. The
    /// `continuing` block is meant to be used to represent structures like the
    /// third expression of a C-style `for` loop head, to which `continue`
    /// statements in the loop's body jump.
    ///
    /// The `continuing` block and its substatements must not contain `Return`
    /// or `Kill` statements, or any `Break` or `Continue` statements targeting
    /// this loop. (It may have `Break` and `Continue` statements targeting
    /// loops or switches nested within the `continuing` block.)
    ///
    /// If present, `break_if` is an expression which is evaluated after the
    /// continuing block. If its value is true, control continues after the
    /// `Loop` statement, rather than branching back to the top of body as
    /// usual. The `break_if` expression corresponds to a "break if" statement
    /// in WGSL, or a loop whose back edge is an `OpBranchConditional`
    /// instruction in SPIR-V.
    ///
    /// [`Break`]: Statement::Break
    /// [`Continue`]: Statement::Continue
    /// [`Kill`]: Statement::Kill
    /// [`Return`]: Statement::Return
    /// [`break if`]: Self::Loop::break_if
    Loop {
        body: Block,
        continuing: Block,
        break_if: Option<Handle<Expression>>,
    },

    /// Exits the innermost enclosing [`Loop`] or [`Switch`].
    ///
    /// A `Break` statement may only appear within a [`Loop`] or [`Switch`]
    /// statement. It may not break out of a [`Loop`] from within the loop's
    /// `continuing` block.
    ///
    /// [`Loop`]: Statement::Loop
    /// [`Switch`]: Statement::Switch
    Break,

    /// Skips to the `continuing` block of the innermost enclosing [`Loop`].
    ///
    /// A `Continue` statement may only appear within the `body` block of the
    /// innermost enclosing [`Loop`] statement. It must not appear within that
    /// loop's `continuing` block.
    ///
    /// [`Loop`]: Statement::Loop
    Continue,

    /// Returns from the function (possibly with a value).
    ///
    /// `Return` statements are forbidden within the `continuing` block of a
    /// [`Loop`] statement.
    ///
    /// [`Loop`]: Statement::Loop
    Return { value: Option<Handle<Expression>> },

    /// Aborts the current shader execution.
    ///
    /// `Kill` statements are forbidden within the `continuing` block of a
    /// [`Loop`] statement.
    ///
    /// [`Loop`]: Statement::Loop
    Kill,

    /// Synchronize invocations within the work group.
    /// The `Barrier` flags control which memory accesses should be synchronized.
    /// If empty, this becomes purely an execution barrier.
    Barrier(Barrier),
    /// Stores a value at an address.
    ///
    /// For [`TypeInner::Atomic`] type behind the pointer, the value
    /// has to be a corresponding scalar.
    /// For other types behind the `pointer<T>`, the value is T.
    ///
    /// This statement is a barrier for any operations on the
    /// `Expression::LocalVariable` or `Expression::GlobalVariable`
    /// that is the destination of an access chain, started
    /// from the `pointer`.
    Store {
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    },
    /// Stores a texel value to an image.
    ///
    /// The `image`, `coordinate`, and `array_index` fields have the same
    /// meanings as the corresponding operands of an [`ImageLoad`] expression;
    /// see that documentation for details. Storing into multisampled images or
    /// images with mipmaps is not supported, so there are no `level` or
    /// `sample` operands.
    ///
    /// This statement is a barrier for any operations on the corresponding
    /// [`Expression::GlobalVariable`] for this image.
    ///
    /// [`ImageLoad`]: Expression::ImageLoad
    ImageStore {
        image: Handle<Expression>,
        coordinate: Handle<Expression>,
        array_index: Option<Handle<Expression>>,
        value: Handle<Expression>,
    },
    /// Atomic function.
    Atomic {
        /// Pointer to an atomic value.
        pointer: Handle<Expression>,
        /// Function to run on the atomic.
        fun: AtomicFunction,
        /// Value to use in the function.
        value: Handle<Expression>,
        /// [`AtomicResult`] expression representing this function's result.
        ///
        /// [`AtomicResult`]: crate::Expression::AtomicResult
        result: Handle<Expression>,
    },
    /// Calls a function.
    ///
    /// If the `result` is `Some`, the corresponding expression has to be
    /// `Expression::CallResult`, and this statement serves as a barrier for any
    /// operations on that expression.
    Call {
        function: Handle<Function>,
        arguments: Vec<Handle<Expression>>,
        result: Option<Handle<Expression>>,
    },
}

/// A function argument.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct FunctionArgument {
    /// Name of the argument, if any.
    pub name: Option<String>,
    /// Type of the argument.
    pub ty: Handle<Type>,
    /// For entry points, an argument has to have a binding
    /// unless it's a structure.
    pub binding: Option<Binding>,
}

/// A function result.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct FunctionResult {
    /// Type of the result.
    pub ty: Handle<Type>,
    /// For entry points, the result has to have a binding
    /// unless it's a structure.
    pub binding: Option<Binding>,
}

/// A function defined in the module.
#[derive(Debug, Default)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct Function {
    /// Name of the function, if any.
    pub name: Option<String>,
    /// Information about function argument.
    pub arguments: Vec<FunctionArgument>,
    /// The result of this function, if any.
    pub result: Option<FunctionResult>,
    /// Local variables defined and used in the function.
    pub local_variables: Arena<LocalVariable>,
    /// Expressions used inside this function.
    ///
    /// An `Expression` must occur before all other `Expression`s that use its
    /// value.
    pub expressions: Arena<Expression>,
    /// Map of expressions that have associated variable names
    pub named_expressions: NamedExpressions,
    /// Block of instructions comprising the body of the function.
    pub body: Block,
}

/// The main function for a pipeline stage.
///
/// An [`EntryPoint`] is a [`Function`] that serves as the main function for a
/// graphics or compute pipeline stage. For example, an `EntryPoint` whose
/// [`stage`] is [`ShaderStage::Vertex`] can serve as a graphics pipeline's
/// vertex shader.
///
/// Since an entry point is called directly by the graphics or compute pipeline,
/// not by other WGSL functions, you must specify what the pipeline should pass
/// as the entry point's arguments, and what values it will return. For example,
/// a vertex shader needs a vertex's attributes as its arguments, but if it's
/// used for instanced draw calls, it will also want to know the instance id.
/// The vertex shader's return value will usually include an output vertex
/// position, and possibly other attributes to be interpolated and passed along
/// to a fragment shader.
///
/// To specify this, the arguments and result of an `EntryPoint`'s [`function`]
/// must each have a [`Binding`], or be structs whose members all have
/// `Binding`s. This associates every value passed to or returned from the entry
/// point with either a [`BuiltIn`] or a [`Location`]:
///
/// -   A [`BuiltIn`] has special semantics, usually specific to its pipeline
///     stage. For example, the result of a vertex shader can include a
///     [`BuiltIn::Position`] value, which determines the position of a vertex
///     of a rendered primitive. Or, a compute shader might take an argument
///     whose binding is [`BuiltIn::WorkGroupSize`], through which the compute
///     pipeline would pass the number of invocations in your workgroup.
///
/// -   A [`Location`] indicates user-defined IO to be passed from one pipeline
///     stage to the next. For example, a vertex shader might also produce a
///     `uv` texture location as a user-defined IO value.
///
/// In other words, the pipeline stage's input and output interface are
/// determined by the bindings of the arguments and result of the `EntryPoint`'s
/// [`function`].
///
/// [`Function`]: crate::Function
/// [`Location`]: Binding::Location
/// [`function`]: EntryPoint::function
/// [`stage`]: EntryPoint::stage
#[derive(Debug)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct EntryPoint {
    /// Name of this entry point, visible externally.
    ///
    /// Entry point names for a given `stage` must be distinct within a module.
    pub name: String,
    /// Shader stage.
    pub stage: ShaderStage,
    /// Early depth test for fragment stages.
    pub early_depth_test: Option<EarlyDepthTest>,
    /// Workgroup size for compute stages
    pub workgroup_size: [u32; 3],
    /// The entrance function.
    pub function: Function,
}

/// Shader module.
///
/// A module is a set of constants, global variables and functions, as well as
/// the types required to define them.
///
/// Some functions are marked as entry points, to be used in a certain shader stage.
///
/// To create a new module, use the `Default` implementation.
/// Alternatively, you can load an existing shader using one of the [available front ends][front].
///
/// When finished, you can export modules using one of the [available backends][back].
#[derive(Debug, Default)]
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
pub struct Module {
    /// Arena for the types defined in this module.
    pub types: UniqueArena<Type>,
    /// Arena for the constants defined in this module.
    pub constants: Arena<Constant>,
    /// Arena for the global variables defined in this module.
    pub global_variables: Arena<GlobalVariable>,
    /// Arena for the functions defined in this module.
    ///
    /// Each function must appear in this arena strictly before all its callers.
    /// Recursion is not supported.
    pub functions: Arena<Function>,
    /// Entry points.
    pub entry_points: Vec<EntryPoint>,
}
