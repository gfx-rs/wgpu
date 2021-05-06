/*! Universal shader translator.

The central structure of the crate is [`Module`].

To improve performance and reduce memory usage, most structures are stored
in an [`Arena`], and can be retrieved using the corresponding [`Handle`].

Functions are described in terms of statement trees. A statement may have
branching control flow, and it may be mutating. Statements refer to expressions.

Expressions form a DAG, without any side effects. Expressions need to be
emitted in order to take effect. This happens in one of the following ways:
  1. Constants and function arguments are implicitly emitted at function start.
      This corresponds to `expression.needs_pre_emit()` condition.
  2. Local and global variables are implicitly emitted. However, in order to use parts
      of them in right-hand-side expressions, the `Expression::Load` must be explicitly emitted,
      with an exception of `StorageClass::Handle` global variables.
  3. Result of `Statement::Call` is automatically emitted.
  4. `Statement::Emit` range is explicitly emitted.

!*/

// TODO: use `strip_prefix` instead when Rust 1.45 <= MSRV
#![allow(
    renamed_and_removed_lints,
    unknown_lints, // requires Rust 1.51
    clippy::new_without_default,
    clippy::unneeded_field_pattern,
    clippy::match_like_matches_macro,
    clippy::manual_strip,
    clippy::unknown_clippy_lints,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    clippy::pattern_type_mismatch
)]
#![deny(clippy::panic)]

mod arena;
pub mod back;
pub mod front;
pub mod proc;
pub mod valid;

pub use crate::arena::{Arena, Handle, Range};

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    num::NonZeroU32,
};

#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

/// Width of a boolean type, in bytes.
pub const BOOL_WIDTH: Bytes = 1;

/// Hash map that is faster but not resilient to DoS attacks.
pub type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;
/// Hash set that is faster but not resilient to DoS attacks.
pub type FastHashSet<K> = HashSet<K, BuildHasherDefault<fxhash::FxHasher>>;

/// Early fragment tests. In a standard situation if a driver determines that it is possible to
/// switch on early depth test it will. Typical situations when early depth test is switched off:
///   - Calling ```discard``` in a shader.
///   - Writing to the depth buffer, unless ConservativeDepth is enabled.
///
/// SPIR-V: ExecutionMode EarlyFragmentTests
/// In GLSL: layout(early_fragment_tests) in;
/// HLSL: Attribute earlydepthstencil
///
/// For more, see:
///   - https://www.khronos.org/opengl/wiki/Early_Fragment_Test#Explicit_specification
///   - https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-earlydepthstencil
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct EarlyDepthTest {
    conservative: Option<ConservativeDepth>,
}
/// Enables adjusting depth without disabling early Z.
///
/// SPIR-V: ExecutionMode DepthGreater/DepthLess/DepthUnchanged
/// GLSL: layout (depth_<greater/less/unchanged/any>) out float gl_FragDepth;
///   - ```depth_any``` option behaves as if the layout qualifier was not present.
/// HLSL: SV_Depth/SV_DepthGreaterEqual/SV_DepthLessEqual
///
/// For more, see:
///   - https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_conservative_depth.txt
///   - https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-semantics#system-value-semantics
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum ConservativeDepth {
    /// Shader may rewrite depth only with a value greater than calculated;
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
#[allow(missing_docs)] // The names are self evident
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// Class of storage for variables.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum StorageClass {
    /// Function locals.
    Function,
    /// Private data, per invocation, mutable.
    Private,
    /// Workgroup shared data, mutable.
    WorkGroup,
    /// Uniform buffer data.
    Uniform,
    /// Storage buffer data, potentially mutable.
    Storage,
    /// Opaque handles, such as samplers and images.
    Handle,
    /// Push constants.
    PushConstant,
}

/// Built-in inputs and outputs.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum BuiltIn {
    Position,
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
    SampleIndex,
    SampleMask,
    // compute
    GlobalInvocationId,
    LocalInvocationId,
    LocalInvocationIndex,
    WorkGroupId,
    WorkGroupSize,
}

/// Number of bytes per scalar.
pub type Bytes = u8;
pub type Alignment = NonZeroU32;

/// Number of components in a vector.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
    #[derive(Default)]
    pub struct StorageAccess: u32 {
        /// Storage can be used as a source for load ops.
        const LOAD = 0x1;
        /// Storage can be used as a target for store ops.
        const STORE = 0x2;
    }
}

// Storage image format.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
pub enum ImageClass {
    /// Regular sampled image.
    Sampled {
        /// Kind of values to sample.
        kind: ScalarKind,
        // Multi-sampled.
        multi: bool,
    },
    /// Depth comparison image.
    Depth,
    /// Storage image.
    Storage(StorageFormat),
}

/// Qualifier of the type level, at which a struct can be used.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum StructLevel {
    /// This is a root level struct, it can't be nested inside
    /// other composite types.
    Root,
    /// This is a normal struct, and it has to be aligned for nesting.
    Normal { alignment: Alignment },
}

/// A data type declared in the module.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Type {
    /// The name of the type, if any.
    pub name: Option<String>,
    /// Inner structure that depends on the kind of the type.
    pub inner: TypeInner,
}

/// Enum with additional information, depending on the kind of type.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
    /// Pointer to another type.
    Pointer {
        base: Handle<Type>,
        class: StorageClass,
    },
    /// Pointer to a value.
    ValuePointer {
        size: Option<VectorSize>,
        kind: ScalarKind,
        width: Bytes,
        class: StorageClass,
    },
    /// Homogenous list of elements.
    Array {
        base: Handle<Type>,
        size: ArraySize,
        stride: u32,
    },
    /// User-defined structure.
    Struct {
        level: StructLevel,
        members: Vec<StructMember>,
        span: u32,
    },
    /// Possibly multidimensional array of texels.
    Image {
        dim: ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
    /// Can be used to sample values from images.
    Sampler { comparison: bool },
}

/// Constant value.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Constant {
    pub name: Option<String>,
    pub specialization: Option<u32>,
    pub inner: ConstantInner,
}

/// A literal scalar value, used in constants.
#[derive(Debug, Clone, Copy, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum Binding {
    /// Built-in shader variable.
    BuiltIn(BuiltIn),
    /// Indexed location.
    Location {
        location: u32,
        interpolation: Option<Interpolation>,
        sampling: Option<Sampling>,
    },
}

/// Pipeline binding information for global resources.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
pub struct GlobalVariable {
    /// Name of the variable, if any.
    pub name: Option<String>,
    /// How this variable is to be stored.
    pub class: StorageClass,
    /// For resources, defines the binding point.
    pub binding: Option<ResourceBinding>,
    /// The type of this variable.
    pub ty: Handle<Type>,
    /// Initial value for this variable.
    pub init: Option<Handle<Constant>>,
    /// Access bit for storage types of images and buffers.
    pub storage_access: StorageAccess,
}

/// Variable defined at function level.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
pub enum UnaryOperator {
    Negate,
    Not,
}

/// Operation that can be applied on two values.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
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

/// Axis on which to compute a derivative.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum DerivativeAxis {
    X,
    Y,
    Width,
}

/// Built-in shader function for testing relation between values.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
pub enum MathFunction {
    // comparison
    Abs,
    Min,
    Max,
    Clamp,
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
}

/// Sampling modifier to control the level of detail.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
    #[derive(Default)]
    pub struct Barrier: u32 {
        /// Barrier affects all `StorageClass::Storage` accesses.
        const STORAGE = 0x1;
        /// Barrier affects all `StorageClass::WorkGroup` accesses.
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
pub enum Expression {
    /// Array access with a computed index.
    Access {
        base: Handle<Expression>,
        index: Handle<Expression>, //int
    },
    /// Array access with a known index.
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
    FunctionArgument(u32),
    /// Reference a global variable.
    GlobalVariable(Handle<GlobalVariable>),
    /// Reference a local variable.
    LocalVariable(Handle<LocalVariable>),
    /// Load a value indirectly.
    Load { pointer: Handle<Expression> },
    /// Sample a point from a sampled or a depth image.
    ImageSample {
        image: Handle<Expression>,
        sampler: Handle<Expression>,
        coordinate: Handle<Expression>,
        array_index: Option<Handle<Expression>>,
        offset: Option<Handle<Constant>>,
        level: SampleLevel,
        depth_ref: Option<Handle<Expression>>,
    },
    /// Load a texel from an image.
    ImageLoad {
        image: Handle<Expression>,
        coordinate: Handle<Expression>,
        array_index: Option<Handle<Expression>>,
        /// For storage images, this is None.
        /// For sampled images, this is the Some(Level).
        /// For multisampled images, this is Some(Sample).
        index: Option<Handle<Expression>>,
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
    },
    /// Cast a simply type to another kind.
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
    Call(Handle<Function>),
    /// Get the length of an array.
    /// The expression must resolve to a pointer to an array with a dynamic size.
    ///
    /// This doesn't match the semantics of spirv's `OpArrayLength`, which must be passed
    /// a pointer to a structure containing a runtime array in its' last field.
    ArrayLength(Handle<Expression>),
}

/// A code block is just a vector of statements.
pub type Block = Vec<Statement>;

/// A case for a switch statement.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct SwitchCase {
    /// Value, upon which the case is considered true.
    pub value: i32,
    /// Body of the cae.
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
pub enum Statement {
    /// Emit a range of expressions, visible to all statements that follow in this block.
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
        default: Block,
    },
    /// Executes a block repeatedly.
    Loop { body: Block, continuing: Block },
    /// Exits the loop.
    Break,
    /// Skips execution to the next iteration of the loop.
    Continue,
    /// Returns from the function (possibly with a value).
    Return { value: Option<Handle<Expression>> },
    /// Aborts the current shader execution.
    Kill,
    /// Synchronize invocations within the work group.
    /// The `Barrier` flags control which memory accesses should be synchronized.
    /// If empty, this becomes purely an execution barrier.
    Barrier(Barrier),
    /// Stores a value at an address.
    ///
    /// This statement is a barrier for any operations on the
    /// `Expression::LocalVariable` or `Expression::GlobalVariable`
    /// that is the destination of an access chain, started
    /// from the `pointer`.
    Store {
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    },
    /// Stores a value to an image.
    ///
    /// Image has to point into a global variable of type `TypeInner::Image`.
    /// This statement is a barrier for any operations on the corresponding
    /// `Expression::GlobalVariable` for this image.
    ImageStore {
        image: Handle<Expression>,
        coordinate: Handle<Expression>,
        array_index: Option<Handle<Expression>>,
        value: Handle<Expression>,
    },
    /// Calls a function.
    ///
    /// If the `result` is `Some`, the corresponding expression has to be
    /// `Expression::Call`, and this statement serves as a barrier for any
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
pub struct FunctionArgument {
    /// Name of the argument, if any.
    pub name: Option<String>,
    /// Type of the argument.
    pub ty: Handle<Type>,
    /// For entry points, an argument has to have a binding
    /// unless it's a structure.
    pub binding: Option<Binding>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct FunctionResult {
    /// Type of the result.
    pub ty: Handle<Type>,
    /// For entry points, the result has to have a binding
    /// unless it's a structure.
    pub binding: Option<Binding>,
}

/// A function defined in the module.
#[derive(Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
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
    pub expressions: Arena<Expression>,
    /// Block of instructions comprising the body of the function.
    pub body: Block,
}

/// Exported function, to be run at a certain stage in the pipeline.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct EntryPoint {
    /// Name of this entry point, visible externally.
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
/// When finished, you can export modules using one of the [available back ends][back].
#[derive(Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Module {
    /// Storage for the types defined in this module.
    pub types: Arena<Type>,
    /// Storage for the constants defined in this module.
    pub constants: Arena<Constant>,
    /// Storage for the global variables defined in this module.
    pub global_variables: Arena<GlobalVariable>,
    /// Storage for the functions defined in this module.
    pub functions: Arena<Function>,
    /// Entry points.
    pub entry_points: Vec<EntryPoint>,
}
