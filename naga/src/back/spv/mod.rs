/*!
Backend for [SPIR-V][spv] (Standard Portable Intermediate Representation).

[spv]: https://www.khronos.org/registry/SPIR-V/
*/

mod block;
mod helpers;
mod image;
mod index;
mod instructions;
mod layout;
mod ray;
mod recyclable;
mod selection;
mod subgroup;
mod writer;

pub use spirv::{Capability, SourceLanguage};

use crate::arena::{Handle, HandleVec};
use crate::proc::{BoundsCheckPolicies, TypeResolution};

use spirv::Word;
use std::ops;
use thiserror::Error;

#[derive(Clone)]
struct PhysicalLayout {
    magic_number: Word,
    version: Word,
    generator: Word,
    bound: Word,
    instruction_schema: Word,
}

#[derive(Default)]
struct LogicalLayout {
    capabilities: Vec<Word>,
    extensions: Vec<Word>,
    ext_inst_imports: Vec<Word>,
    memory_model: Vec<Word>,
    entry_points: Vec<Word>,
    execution_modes: Vec<Word>,
    debugs: Vec<Word>,
    annotations: Vec<Word>,
    declarations: Vec<Word>,
    function_declarations: Vec<Word>,
    function_definitions: Vec<Word>,
}

struct Instruction {
    op: spirv::Op,
    wc: u32,
    type_id: Option<Word>,
    result_id: Option<Word>,
    operands: Vec<Word>,
}

const BITS_PER_BYTE: crate::Bytes = 8;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("The requested entry point couldn't be found")]
    EntryPointNotFound,
    #[error("target SPIRV-{0}.{1} is not supported")]
    UnsupportedVersion(u8, u8),
    #[error("using {0} requires at least one of the capabilities {1:?}, but none are available")]
    MissingCapabilities(&'static str, Vec<Capability>),
    #[error("unimplemented {0}")]
    FeatureNotImplemented(&'static str),
    #[error("module is not validated properly: {0}")]
    Validation(&'static str),
    #[error("overrides should not be present at this stage")]
    Override,
}

#[derive(Default)]
struct IdGenerator(Word);

impl IdGenerator {
    fn next(&mut self) -> Word {
        self.0 += 1;
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct DebugInfo<'a> {
    pub source_code: &'a str,
    pub file_name: &'a std::path::Path,
    pub language: SourceLanguage,
}

/// A SPIR-V block to which we are still adding instructions.
///
/// A `Block` represents a SPIR-V block that does not yet have a termination
/// instruction like `OpBranch` or `OpReturn`.
///
/// The `OpLabel` that starts the block is implicit. It will be emitted based on
/// `label_id` when we write the block to a `LogicalLayout`.
///
/// To terminate a `Block`, pass the block and the termination instruction to
/// `Function::consume`. This takes ownership of the `Block` and transforms it
/// into a `TerminatedBlock`.
struct Block {
    label_id: Word,
    body: Vec<Instruction>,
}

/// A SPIR-V block that ends with a termination instruction.
struct TerminatedBlock {
    label_id: Word,
    body: Vec<Instruction>,
}

impl Block {
    const fn new(label_id: Word) -> Self {
        Block {
            label_id,
            body: Vec::new(),
        }
    }
}

struct LocalVariable {
    id: Word,
    instruction: Instruction,
}

struct ResultMember {
    id: Word,
    type_id: Word,
    built_in: Option<crate::BuiltIn>,
}

struct EntryPointContext {
    argument_ids: Vec<Word>,
    results: Vec<ResultMember>,
}

#[derive(Default)]
struct Function {
    signature: Option<Instruction>,
    parameters: Vec<FunctionArgument>,
    variables: crate::FastHashMap<Handle<crate::LocalVariable>, LocalVariable>,

    /// A map taking an expression that yields a composite value (array, matrix)
    /// to the temporary variables we have spilled it to, if any. Spilling
    /// allows us to render an arbitrary chain of [`Access`] and [`AccessIndex`]
    /// expressions as an `OpAccessChain` and an `OpLoad` (plus bounds checks).
    /// This supports dynamic indexing of by-value arrays and matrices, which
    /// SPIR-V does not.
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    spilled_composites: crate::FastIndexMap<Handle<crate::Expression>, LocalVariable>,

    /// A set of expressions that are either in [`spilled_composites`] or refer
    /// to some component/element of such.
    ///
    /// [`spilled_composites`]: Function::spilled_composites
    spilled_accesses: crate::arena::HandleSet<crate::Expression>,

    /// A map taking each expression to the number of [`Access`] and
    /// [`AccessIndex`] expressions that uses it as a base value. If an
    /// expression has no entry, its count is zero: it is never used as a
    /// [`Access`] or [`AccessIndex`] base.
    ///
    /// We use this, together with [`ExpressionInfo::ref_count`], to recognize
    /// the tips of chains of [`Access`] and [`AccessIndex`] expressions that
    /// access spilled values --- expressions in [`spilled_composites`]. We
    /// defer generating code for the chain until we reach its tip, so we can
    /// handle it with a single instruction.
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`AccessIndex`]: crate::Expression::AccessIndex
    /// [`ExpressionInfo::ref_count`]: crate::valid::ExpressionInfo
    /// [`spilled_composites`]: Function::spilled_composites
    access_uses: crate::FastHashMap<Handle<crate::Expression>, usize>,

    blocks: Vec<TerminatedBlock>,
    entry_point_context: Option<EntryPointContext>,
}

impl Function {
    fn consume(&mut self, mut block: Block, termination: Instruction) {
        block.body.push(termination);
        self.blocks.push(TerminatedBlock {
            label_id: block.label_id,
            body: block.body,
        })
    }

    fn parameter_id(&self, index: u32) -> Word {
        match self.entry_point_context {
            Some(ref context) => context.argument_ids[index as usize],
            None => self.parameters[index as usize]
                .instruction
                .result_id
                .unwrap(),
        }
    }
}

/// Characteristics of a SPIR-V `OpTypeImage` type.
///
/// SPIR-V requires non-composite types to be unique, including images. Since we
/// use `LocalType` for this deduplication, it's essential that `LocalImageType`
/// be equal whenever the corresponding `OpTypeImage`s would be. To reduce the
/// likelihood of mistakes, we use fields that correspond exactly to the
/// operands of an `OpTypeImage` instruction, using the actual SPIR-V types
/// where practical.
#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
struct LocalImageType {
    sampled_type: crate::Scalar,
    dim: spirv::Dim,
    flags: ImageTypeFlags,
    image_format: spirv::ImageFormat,
}

bitflags::bitflags! {
    /// Flags corresponding to the boolean(-ish) parameters to OpTypeImage.
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    pub struct ImageTypeFlags: u8 {
        const DEPTH = 0x1;
        const ARRAYED = 0x2;
        const MULTISAMPLED = 0x4;
        const SAMPLED = 0x8;
    }
}

impl LocalImageType {
    /// Construct a `LocalImageType` from the fields of a `TypeInner::Image`.
    fn from_inner(dim: crate::ImageDimension, arrayed: bool, class: crate::ImageClass) -> Self {
        let make_flags = |multi: bool, other: ImageTypeFlags| -> ImageTypeFlags {
            let mut flags = other;
            flags.set(ImageTypeFlags::ARRAYED, arrayed);
            flags.set(ImageTypeFlags::MULTISAMPLED, multi);
            flags
        };

        let dim = spirv::Dim::from(dim);

        match class {
            crate::ImageClass::Sampled { kind, multi } => LocalImageType {
                sampled_type: crate::Scalar { kind, width: 4 },
                dim,
                flags: make_flags(multi, ImageTypeFlags::SAMPLED),
                image_format: spirv::ImageFormat::Unknown,
            },
            crate::ImageClass::Depth { multi } => LocalImageType {
                sampled_type: crate::Scalar {
                    kind: crate::ScalarKind::Float,
                    width: 4,
                },
                dim,
                flags: make_flags(multi, ImageTypeFlags::DEPTH | ImageTypeFlags::SAMPLED),
                image_format: spirv::ImageFormat::Unknown,
            },
            crate::ImageClass::Storage { format, access: _ } => LocalImageType {
                sampled_type: format.into(),
                dim,
                flags: make_flags(false, ImageTypeFlags::empty()),
                image_format: format.into(),
            },
        }
    }
}

/// A numeric type, for use in [`LocalType`].
#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum NumericType {
    Scalar(crate::Scalar),
    Vector {
        size: crate::VectorSize,
        scalar: crate::Scalar,
    },
    Matrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        scalar: crate::Scalar,
    },
}

impl NumericType {
    const fn from_inner(inner: &crate::TypeInner) -> Option<Self> {
        match *inner {
            crate::TypeInner::Scalar(scalar) | crate::TypeInner::Atomic(scalar) => {
                Some(NumericType::Scalar(scalar))
            }
            crate::TypeInner::Vector { size, scalar } => Some(NumericType::Vector { size, scalar }),
            crate::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Some(NumericType::Matrix {
                columns,
                rows,
                scalar,
            }),
            _ => None,
        }
    }
}

/// A SPIR-V type constructed during code generation.
///
/// This is the variant of [`LookupType`] used to represent types that might not
/// be available in the arena. Variants are present here for one of two reasons:
///
/// -   They represent types synthesized during code generation, as explained
///     in the documentation for [`LookupType`].
///
/// -   They represent types for which SPIR-V forbids duplicate `OpType...`
///     instructions, requiring deduplication.
///
/// This is not a complete copy of [`TypeInner`]: for example, SPIR-V generation
/// never synthesizes new struct types, so `LocalType` has nothing for that.
///
/// Each `LocalType` variant should be handled identically to its analogous
/// `TypeInner` variant. You can use the [`LocalType::from_inner`] function to
/// help with this, by converting everything possible to a `LocalType` before
/// inspecting it.
///
/// ## `LocalType` equality and SPIR-V `OpType` uniqueness
///
/// The definition of `Eq` on `LocalType` is carefully chosen to help us follow
/// certain SPIR-V rules. SPIR-V §2.8 requires some classes of `OpType...`
/// instructions to be unique; for example, you can't have two `OpTypeInt 32 1`
/// instructions in the same module. All 32-bit signed integers must use the
/// same type id.
///
/// All SPIR-V types that must be unique can be represented as a `LocalType`,
/// and two `LocalType`s are always `Eq` if SPIR-V would require them to use the
/// same `OpType...` instruction. This lets us avoid duplicates by recording the
/// ids of the type instructions we've already generated in a hash table,
/// [`Writer::lookup_type`], keyed by `LocalType`.
///
/// As another example, [`LocalImageType`], stored in the `LocalType::Image`
/// variant, is designed to help us deduplicate `OpTypeImage` instructions. See
/// its documentation for details.
///
/// `LocalType` also includes variants like `Pointer` that do not need to be
/// unique - but it is harmless to avoid the duplication.
///
/// As it always must, the `Hash` implementation respects the `Eq` relation.
///
/// [`TypeInner`]: crate::TypeInner
#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LocalType {
    /// A numeric type.
    Numeric(NumericType),
    LocalPointer {
        base: NumericType,
        class: spirv::StorageClass,
    },
    Pointer {
        base: Handle<crate::Type>,
        class: spirv::StorageClass,
    },
    Image(LocalImageType),
    SampledImage {
        image_type_id: Word,
    },
    Sampler,
    /// Equivalent to a [`LocalType::Pointer`] whose `base` is a Naga IR [`BindingArray`]. SPIR-V
    /// permits duplicated `OpTypePointer` ids, so it's fine to have two different [`LocalType`]
    /// representations for pointer types.
    ///
    /// [`BindingArray`]: crate::TypeInner::BindingArray
    PointerToBindingArray {
        base: Handle<crate::Type>,
        size: u32,
        space: crate::AddressSpace,
    },
    BindingArray {
        base: Handle<crate::Type>,
        size: u32,
    },
    AccelerationStructure,
    RayQuery,
}

/// A type encountered during SPIR-V generation.
///
/// In the process of writing SPIR-V, we need to synthesize various types for
/// intermediate results and such: pointer types, vector/matrix component types,
/// or even booleans, which usually appear in SPIR-V code even when they're not
/// used by the module source.
///
/// However, we can't use `crate::Type` or `crate::TypeInner` for these, as the
/// type arena may not contain what we need (it only contains types used
/// directly by other parts of the IR), and the IR module is immutable, so we
/// can't add anything to it.
///
/// So for local use in the SPIR-V writer, we use this type, which holds either
/// a handle into the arena, or a [`LocalType`] containing something synthesized
/// locally.
///
/// This is very similar to the [`proc::TypeResolution`] enum, with `LocalType`
/// playing the role of `TypeInner`. However, `LocalType` also has other
/// properties needed for SPIR-V generation; see the description of
/// [`LocalType`] for details.
///
/// [`proc::TypeResolution`]: crate::proc::TypeResolution
#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LookupType {
    Handle(Handle<crate::Type>),
    Local(LocalType),
}

impl From<LocalType> for LookupType {
    fn from(local: LocalType) -> Self {
        Self::Local(local)
    }
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
struct LookupFunctionType {
    parameter_type_ids: Vec<Word>,
    return_type_id: Word,
}

impl LocalType {
    fn from_inner(inner: &crate::TypeInner) -> Option<Self> {
        Some(match *inner {
            crate::TypeInner::Scalar(_)
            | crate::TypeInner::Atomic(_)
            | crate::TypeInner::Vector { .. }
            | crate::TypeInner::Matrix { .. } => {
                // We expect `NumericType::from_inner` to handle all
                // these cases, so unwrap.
                LocalType::Numeric(NumericType::from_inner(inner).unwrap())
            }
            crate::TypeInner::Pointer { base, space } => LocalType::Pointer {
                base,
                class: helpers::map_storage_class(space),
            },
            crate::TypeInner::ValuePointer {
                size: Some(size),
                scalar,
                space,
            } => LocalType::LocalPointer {
                base: NumericType::Vector { size, scalar },
                class: helpers::map_storage_class(space),
            },
            crate::TypeInner::ValuePointer {
                size: None,
                scalar,
                space,
            } => LocalType::LocalPointer {
                base: NumericType::Scalar(scalar),
                class: helpers::map_storage_class(space),
            },
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => LocalType::Image(LocalImageType::from_inner(dim, arrayed, class)),
            crate::TypeInner::Sampler { comparison: _ } => LocalType::Sampler,
            crate::TypeInner::AccelerationStructure => LocalType::AccelerationStructure,
            crate::TypeInner::RayQuery => LocalType::RayQuery,
            crate::TypeInner::Array { .. }
            | crate::TypeInner::Struct { .. }
            | crate::TypeInner::BindingArray { .. } => return None,
        })
    }
}

#[derive(Debug)]
enum Dimension {
    Scalar,
    Vector,
    Matrix,
}

/// A map from evaluated [`Expression`](crate::Expression)s to their SPIR-V ids.
///
/// When we emit code to evaluate a given `Expression`, we record the
/// SPIR-V id of its value here, under its `Handle<Expression>` index.
///
/// A `CachedExpressions` value can be indexed by a `Handle<Expression>` value.
///
/// [emit]: index.html#expression-evaluation-time-and-scope
#[derive(Default)]
struct CachedExpressions {
    ids: HandleVec<crate::Expression, Word>,
}
impl CachedExpressions {
    fn reset(&mut self, length: usize) {
        self.ids.clear();
        self.ids.resize(length, 0);
    }
}
impl ops::Index<Handle<crate::Expression>> for CachedExpressions {
    type Output = Word;
    fn index(&self, h: Handle<crate::Expression>) -> &Word {
        let id = &self.ids[h];
        if *id == 0 {
            unreachable!("Expression {:?} is not cached!", h);
        }
        id
    }
}
impl ops::IndexMut<Handle<crate::Expression>> for CachedExpressions {
    fn index_mut(&mut self, h: Handle<crate::Expression>) -> &mut Word {
        let id = &mut self.ids[h];
        if *id != 0 {
            unreachable!("Expression {:?} is already cached!", h);
        }
        id
    }
}
impl recyclable::Recyclable for CachedExpressions {
    fn recycle(self) -> Self {
        CachedExpressions {
            ids: self.ids.recycle(),
        }
    }
}

#[derive(Eq, Hash, PartialEq)]
enum CachedConstant {
    Literal(crate::proc::HashableLiteral),
    Composite {
        ty: LookupType,
        constituent_ids: Vec<Word>,
    },
    ZeroValue(Word),
}

/// The SPIR-V representation of a [`crate::GlobalVariable`].
///
/// In the Vulkan spec 1.3.296, the section [Descriptor Set Interface][dsi] says:
///
/// > Variables identified with the `Uniform` storage class are used to access
/// > transparent buffer backed resources. Such variables *must* be:
/// >
/// > -   typed as `OpTypeStruct`, or an array of this type,
/// >
/// > -   identified with a `Block` or `BufferBlock` decoration, and
/// >
/// > -   laid out explicitly using the `Offset`, `ArrayStride`, and `MatrixStride`
/// >     decorations as specified in "Offset and Stride Assignment".
///
/// This is followed by identical language for the `StorageBuffer`,
/// except that a `BufferBlock` decoration is not allowed.
///
/// When we encounter a global variable in the [`Storage`] or [`Uniform`]
/// address spaces whose type is not already [`Struct`], this backend implicitly
/// wraps the global variable in a struct: we generate a SPIR-V global variable
/// holding an `OpTypeStruct` with a single member, whose type is what the Naga
/// global's type would suggest, decorated as required above.
///
/// The [`helpers::global_needs_wrapper`] function determines whether a given
/// [`crate::GlobalVariable`] needs to be wrapped.
///
/// [dsi]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-resources-descset
/// [`Storage`]: crate::AddressSpace::Storage
/// [`Uniform`]: crate::AddressSpace::Uniform
/// [`Struct`]: crate::TypeInner::Struct
#[derive(Clone)]
struct GlobalVariable {
    /// The SPIR-V id of the `OpVariable` that declares the global.
    ///
    /// If this global has been implicitly wrapped in an `OpTypeStruct`, this id
    /// refers to the wrapper, not the original Naga value it contains. If you
    /// need the Naga value, use [`access_id`] instead of this field.
    ///
    /// If this global is not implicitly wrapped, this is the same as
    /// [`access_id`].
    ///
    /// This is used to compute the `access_id` pointer in function prologues,
    /// and used for `ArrayLength` expressions, which need to pass the wrapper
    /// struct.
    ///
    /// [`access_id`]: GlobalVariable::access_id
    var_id: Word,

    /// The loaded value of a `AddressSpace::Handle` global variable.
    ///
    /// If the current function uses this global variable, this is the id of an
    /// `OpLoad` instruction in the function's prologue that loads its value.
    /// (This value is assigned as we write the prologue code of each function.)
    /// It is then used for all operations on the global, such as `OpImageSample`.
    handle_id: Word,

    /// The SPIR-V id of a pointer to this variable's Naga IR value.
    ///
    /// If the current function uses this global variable, and it has been
    /// implicitly wrapped in an `OpTypeStruct`, this is the id of an
    /// `OpAccessChain` instruction in the function's prologue that refers to
    /// the wrapped value inside the struct. (This value is assigned as we write
    /// the prologue code of each function.) If you need the wrapper struct
    /// itself, use [`var_id`] instead of this field.
    ///
    /// If this global is not implicitly wrapped, this is the same as
    /// [`var_id`].
    ///
    /// [`var_id`]: GlobalVariable::var_id
    access_id: Word,
}

impl GlobalVariable {
    const fn dummy() -> Self {
        Self {
            var_id: 0,
            handle_id: 0,
            access_id: 0,
        }
    }

    const fn new(id: Word) -> Self {
        Self {
            var_id: id,
            handle_id: 0,
            access_id: 0,
        }
    }

    /// Prepare `self` for use within a single function.
    fn reset_for_function(&mut self) {
        self.handle_id = 0;
        self.access_id = 0;
    }
}

struct FunctionArgument {
    /// Actual instruction of the argument.
    instruction: Instruction,
    handle_id: Word,
}

/// Tracks the expressions for which the backend emits the following instructions:
/// - OpConstantTrue
/// - OpConstantFalse
/// - OpConstant
/// - OpConstantComposite
/// - OpConstantNull
struct ExpressionConstnessTracker {
    inner: crate::arena::HandleSet<crate::Expression>,
}

impl ExpressionConstnessTracker {
    fn from_arena(arena: &crate::Arena<crate::Expression>) -> Self {
        let mut inner = crate::arena::HandleSet::for_arena(arena);
        for (handle, expr) in arena.iter() {
            let insert = match *expr {
                crate::Expression::Literal(_)
                | crate::Expression::ZeroValue(_)
                | crate::Expression::Constant(_) => true,
                crate::Expression::Compose { ref components, .. } => {
                    components.iter().all(|&h| inner.contains(h))
                }
                crate::Expression::Splat { value, .. } => inner.contains(value),
                _ => false,
            };
            if insert {
                inner.insert(handle);
            }
        }
        Self { inner }
    }

    fn is_const(&self, value: Handle<crate::Expression>) -> bool {
        self.inner.contains(value)
    }
}

/// General information needed to emit SPIR-V for Naga statements.
struct BlockContext<'w> {
    /// The writer handling the module to which this code belongs.
    writer: &'w mut Writer,

    /// The [`Module`](crate::Module) for which we're generating code.
    ir_module: &'w crate::Module,

    /// The [`Function`](crate::Function) for which we're generating code.
    ir_function: &'w crate::Function,

    /// Information module validation produced about
    /// [`ir_function`](BlockContext::ir_function).
    fun_info: &'w crate::valid::FunctionInfo,

    /// The [`spv::Function`](Function) to which we are contributing SPIR-V instructions.
    function: &'w mut Function,

    /// SPIR-V ids for expressions we've evaluated.
    cached: CachedExpressions,

    /// The `Writer`'s temporary vector, for convenience.
    temp_list: Vec<Word>,

    /// Tracks the constness of `Expression`s residing in `self.ir_function.expressions`
    expression_constness: ExpressionConstnessTracker,
}

impl BlockContext<'_> {
    fn gen_id(&mut self) -> Word {
        self.writer.id_gen.next()
    }

    fn get_type_id(&mut self, lookup_type: LookupType) -> Word {
        self.writer.get_type_id(lookup_type)
    }

    fn get_expression_type_id(&mut self, tr: &TypeResolution) -> Word {
        self.writer.get_expression_type_id(tr)
    }

    fn get_index_constant(&mut self, index: Word) -> Word {
        self.writer.get_constant_scalar(crate::Literal::U32(index))
    }

    fn get_scope_constant(&mut self, scope: Word) -> Word {
        self.writer
            .get_constant_scalar(crate::Literal::I32(scope as _))
    }

    fn get_pointer_id(&mut self, handle: Handle<crate::Type>, class: spirv::StorageClass) -> Word {
        self.writer.get_pointer_id(handle, class)
    }
}

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_gen: IdGenerator,

    /// The set of capabilities modules are permitted to use.
    ///
    /// This is initialized from `Options::capabilities`.
    capabilities_available: Option<crate::FastHashSet<Capability>>,

    /// The set of capabilities used by this module.
    ///
    /// If `capabilities_available` is `Some`, then this is always a subset of
    /// that.
    capabilities_used: crate::FastIndexSet<Capability>,

    /// The set of spirv extensions used.
    extensions_used: crate::FastIndexSet<&'static str>,

    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    flags: WriterFlags,
    bounds_check_policies: BoundsCheckPolicies,
    zero_initialize_workgroup_memory: ZeroInitializeWorkgroupMemoryMode,
    void_type: Word,
    //TODO: convert most of these into vectors, addressable by handle indices
    lookup_type: crate::FastHashMap<LookupType, Word>,
    lookup_function: crate::FastHashMap<Handle<crate::Function>, Word>,
    lookup_function_type: crate::FastHashMap<LookupFunctionType, Word>,
    /// Indexed by const-expression handle indexes
    constant_ids: HandleVec<crate::Expression, Word>,
    cached_constants: crate::FastHashMap<CachedConstant, Word>,
    global_variables: HandleVec<crate::GlobalVariable, GlobalVariable>,
    binding_map: BindingMap,

    // Cached expressions are only meaningful within a BlockContext, but we
    // retain the table here between functions to save heap allocations.
    saved_cached: CachedExpressions,

    gl450_ext_inst_id: Word,

    // Just a temporary list of SPIR-V ids
    temp_list: Vec<Word>,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WriterFlags: u32 {
        /// Include debug labels for everything.
        const DEBUG = 0x1;

        /// Flip Y coordinate of [`BuiltIn::Position`] output.
        ///
        /// [`BuiltIn::Position`]: crate::BuiltIn::Position
        const ADJUST_COORDINATE_SPACE = 0x2;

        /// Emit [`OpName`][op] for input/output locations.
        ///
        /// Contrary to spec, some drivers treat it as semantic, not allowing
        /// any conflicts.
        ///
        /// [op]: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpName
        const LABEL_VARYINGS = 0x4;

        /// Emit [`PointSize`] output builtin to vertex shaders, which is
        /// required for drawing with `PointList` topology.
        ///
        /// [`PointSize`]: crate::BuiltIn::PointSize
        const FORCE_POINT_SIZE = 0x8;

        /// Clamp [`BuiltIn::FragDepth`] output between 0 and 1.
        ///
        /// [`BuiltIn::FragDepth`]: crate::BuiltIn::FragDepth
        const CLAMP_FRAG_DEPTH = 0x10;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindingInfo {
    /// If the binding is an unsized binding array, this overrides the size.
    pub binding_array_size: Option<u32>,
}

// Using `BTreeMap` instead of `HashMap` so that we can hash itself.
pub type BindingMap = std::collections::BTreeMap<crate::ResourceBinding, BindingInfo>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroInitializeWorkgroupMemoryMode {
    /// Via `VK_KHR_zero_initialize_workgroup_memory` or Vulkan 1.3
    Native,
    /// Via assignments + barrier
    Polyfill,
    None,
}

#[derive(Debug, Clone)]
pub struct Options<'a> {
    /// (Major, Minor) target version of the SPIR-V.
    pub lang_version: (u8, u8),

    /// Configuration flags for the writer.
    pub flags: WriterFlags,

    /// Map of resources to information about the binding.
    pub binding_map: BindingMap,

    /// If given, the set of capabilities modules are allowed to use. Code that
    /// requires capabilities beyond these is rejected with an error.
    ///
    /// If this is `None`, all capabilities are permitted.
    pub capabilities: Option<crate::FastHashSet<Capability>>,

    /// How should generate code handle array, vector, matrix, or image texel
    /// indices that are out of range?
    pub bounds_check_policies: BoundsCheckPolicies,

    /// Dictates the way workgroup variables should be zero initialized
    pub zero_initialize_workgroup_memory: ZeroInitializeWorkgroupMemoryMode,

    pub debug_info: Option<DebugInfo<'a>>,
}

impl<'a> Default for Options<'a> {
    fn default() -> Self {
        let mut flags = WriterFlags::ADJUST_COORDINATE_SPACE
            | WriterFlags::LABEL_VARYINGS
            | WriterFlags::CLAMP_FRAG_DEPTH;
        if cfg!(debug_assertions) {
            flags |= WriterFlags::DEBUG;
        }
        Options {
            lang_version: (1, 0),
            flags,
            binding_map: BindingMap::default(),
            capabilities: None,
            bounds_check_policies: BoundsCheckPolicies::default(),
            zero_initialize_workgroup_memory: ZeroInitializeWorkgroupMemoryMode::Polyfill,
            debug_info: None,
        }
    }
}

// A subset of options meant to be changed per pipeline.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct PipelineOptions {
    /// The stage of the entry point.
    pub shader_stage: crate::ShaderStage,
    /// The name of the entry point.
    ///
    /// If no entry point that matches is found while creating a [`Writer`], a error will be thrown.
    pub entry_point: String,
}

pub fn write_vec(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
    options: &Options,
    pipeline_options: Option<&PipelineOptions>,
) -> Result<Vec<u32>, Error> {
    let mut words: Vec<u32> = Vec::new();
    let mut w = Writer::new(options)?;

    w.write(
        module,
        info,
        pipeline_options,
        &options.debug_info,
        &mut words,
    )?;
    Ok(words)
}
