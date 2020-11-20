//! Universal shader translator.
//!
//! The central structure of the crate is [`Module`].
//!
//! To improve performance and reduce memory usage, most structures are stored
//! in an [`Arena`], and can be retrieved using the corresponding [`Handle`].
#![allow(
    clippy::new_without_default,
    clippy::unneeded_field_pattern,
    clippy::match_like_matches_macro
)]
// TODO: use `strip_prefix` instead when Rust 1.45 <= MSRV
#![allow(clippy::manual_strip, clippy::unknown_clippy_lints)]
#![deny(clippy::panic)]

mod arena;
pub mod back;
pub mod front;
pub mod proc;

pub use crate::arena::{Arena, Handle};

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    num::NonZeroU32,
};

#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

/// Hash map that is faster but not resilient to DoS attacks.
pub type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;
/// Hash set that is faster but not resilient to DoS attacks.
pub type FastHashSet<K> = HashSet<K, BuildHasherDefault<fxhash::FxHasher>>;

/// Metadata for a given module.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Header {
    /// Major, minor and patch version.
    ///
    /// Currently used only for the SPIR-V back end.
    pub version: (u8, u8, u8),
    /// Magic number identifying the tool that generated the shader code.
    ///
    /// Can safely be set to 0.
    pub generator: u32,
}

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
#[allow(missing_docs)] // The names are self evident
pub enum StorageClass {
    /// Function locals.
    Function,
    /// Pipeline input, per invocation.
    Input,
    /// Pipeline output, per invocation, mutable.
    Output,
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
    // vertex
    BaseInstance,
    BaseVertex,
    ClipDistance,
    InstanceIndex,
    Position,
    VertexIndex,
    // fragment
    PointSize,
    FragCoord,
    FrontFacing,
    SampleIndex,
    FragDepth,
    // compute
    GlobalInvocationId,
    LocalInvocationId,
    LocalInvocationIndex,
    WorkGroupId,
}

/// Number of bytes.
pub type Bytes = u8;

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

/// Describes where a struct member is placed.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum MemberOrigin {
    /// Member is local to the shader.
    Empty,
    /// Built-in shader variable.
    BuiltIn(BuiltIn),
    /// Offset within the struct.
    Offset(u32),
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
    /// Indicates a tessellation patch.
    Patch,
    /// When used with multi-sampling rasterization, allow
    /// a single interpolation location for an entire pixel.
    Centroid,
    /// When used with multi-sampling rasterization, require
    /// per-sample interpolation.
    Sample,
}

/// Member of a user-defined structure.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct StructMember {
    pub name: Option<String>,
    pub origin: MemberOrigin,
    pub ty: Handle<Type>,
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
    /// Pointer to a value.
    Pointer {
        base: Handle<Type>,
        class: StorageClass,
    },
    /// Homogenous list of elements.
    Array {
        base: Handle<Type>,
        size: ArraySize,
        stride: Option<NonZeroU32>,
    },
    /// User-defined structure.
    Struct { members: Vec<StructMember> },
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
    pub ty: Handle<Type>,
}

/// Additional information, dependendent on the kind of constant.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum ConstantInner {
    Sint(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
    Composite(Vec<Handle<Constant>>),
}

/// Describes how an input/output variable is to be bound.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum Binding {
    /// Built-in shader variable.
    BuiltIn(BuiltIn),
    /// Indexed location.
    Location(u32),
    /// Binding within a resource group.
    Resource { group: u32, binding: u32 },
}

bitflags::bitflags! {
    /// Indicates how a global variable is used.
    #[cfg_attr(feature = "serialize", derive(Serialize))]
    #[cfg_attr(feature = "deserialize", derive(Deserialize))]
    pub struct GlobalUse: u8 {
        /// Data will be read from the variable.
        const LOAD = 0x1;
        /// Data will be written to the variable.
        const STORE = 0x2;
    }
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
    /// How this variable is to be bound.
    pub binding: Option<Binding>,
    /// The type of this variable.
    pub ty: Handle<Type>,
    /// Initial value for this variable.
    pub init: Option<Handle<Constant>>,
    /// The interpolation qualifier, if any.
    /// If the this `GlobalVariable` is a vertex output
    /// or fragment input, `None` corresponds to the
    /// `smooth`/`perspective` interpolation qualifier.
    pub interpolation: Option<Interpolation>,
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

/// Built-in shader function.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum IntrinsicFunction {
    Any,
    All,
    IsNan,
    IsInf,
    IsFinite,
    IsNormal,
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

/// Origin of a function to call.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum FunctionOrigin {
    Local(Handle<Function>),
    // External {
    //     namespace: String, // Maybe this should be a handle to a namespace Arena?
    //     function: String,
    // },
    External(String),
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
}

/// An expression that can be evaluated to obtain a value.
#[derive(Clone, Debug)]
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
        level: SampleLevel,
        depth_ref: Option<Handle<Expression>>,
    },
    /// Load a texel from an image.
    ImageLoad {
        image: Handle<Expression>,
        coordinate: Handle<Expression>,
        /// For storage images, this is None.
        /// For sampled images, this is the Some(Level).
        /// For multisampled images, this is Some(Sample).
        index: Option<Handle<Expression>>,
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
    /// Call an intrinsic function.
    Intrinsic {
        fun: IntrinsicFunction,
        argument: Handle<Expression>,
    },
    /// Transpose of a matrix.
    Transpose(Handle<Expression>),
    /// Dot product between two vectors.
    DotProduct(Handle<Expression>, Handle<Expression>),
    /// Cross product between two vectors.
    CrossProduct(Handle<Expression>, Handle<Expression>),
    /// Cast a simply type to another kind.
    As {
        /// Source expression, which can only be a scalar or a vector.
        expr: Handle<Expression>,
        /// Target scalar kind.
        kind: ScalarKind,
        /// True = conversion needs to take place; False = bitcast.
        convert: bool,
    },
    /// Compute the derivative on an axis.
    Derivative {
        axis: DerivativeAxis,
        //modifier,
        expr: Handle<Expression>,
    },
    /// Call another function.
    Call {
        origin: FunctionOrigin,
        arguments: Vec<Handle<Expression>>,
    },
    /// Get the length of an array.
    ArrayLength(Handle<Expression>),
}

/// A code block is just a vector of statements.
pub type Block = Vec<Statement>;

/// Marker type, used for falling through in a switch statement.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct FallThrough;

/// Instructions which make up an executable block.
// Clone is used only for error reporting and is not intended for end users
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum Statement {
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
        cases: FastHashMap<i32, (Block, Option<FallThrough>)>,
        default: Block,
    },
    /// Executes a block repeatedly.
    Loop { body: Block, continuing: Block },
    //TODO: move terminator variations into a separate enum?
    /// Exits the loop.
    Break,
    /// Skips execution to the next iteration of the loop.
    Continue,
    /// Returns from the function (possibly with a value).
    Return { value: Option<Handle<Expression>> },
    /// Aborts the current shader execution.
    Kill,
    /// Stores a value at an address.
    Store {
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    },
}

/// A function argument.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct FunctionArgument {
    /// Name of the argument, if any.
    pub name: Option<String>,
    /// Type of the argument.
    pub ty: Handle<Type>,
}

/// A function defined in the module.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Function {
    /// Name of the function, if any.
    pub name: Option<String>,
    /// Information about function argument.
    pub arguments: Vec<FunctionArgument>,
    /// The return type of this function, if any.
    pub return_type: Option<Handle<Type>>,
    /// Vector of global variable usages.
    ///
    /// Each item corresponds to a global variable in the module.
    pub global_usage: Vec<GlobalUse>,
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
/// To create a new module, use [`Module::from_header`] or [`Module::generate_empty`].
/// Alternatively, you can load an existing shader using one of the [available front ends][front].
///
/// When finished, you can export modules using one of the [available back ends][back].
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct Module {
    /// Header containing module metadata.
    pub header: Header,
    /// Storage for the types defined in this module.
    pub types: Arena<Type>,
    /// Storage for the constants defined in this module.
    pub constants: Arena<Constant>,
    /// Storage for the global variables defined in this module.
    pub global_variables: Arena<GlobalVariable>,
    /// Storage for the functions defined in this module.
    pub functions: Arena<Function>,
    /// Exported entry points.
    pub entry_points: FastHashMap<(ShaderStage, String), EntryPoint>,
}
