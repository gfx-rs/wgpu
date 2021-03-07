use super::{
    analyzer::{Analysis, AnalysisError, FunctionInfo, GlobalUse},
    typifier::{ResolveContext, Typifier, TypifyError},
};
use crate::{
    arena::{Arena, Handle},
    FastHashSet,
};
use bit_set::BitSet;
use thiserror::Error;

const MAX_WORKGROUP_SIZE: u32 = 0x4000;

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct TypeFlags: u8 {
        /// Can be used for data variables.
        const DATA = 0x1;
        /// The data type has known size.
        const SIZED = 0x2;
        /// Can be be used for interfacing between pipeline stages.
        const INTERFACE = 0x4;
        /// Can be used for host-shareable structures.
        const HOST_SHARED = 0x8;
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct BlockFlags: u8 {
        /// The control can jump out of this block.
        const CAN_JUMP = 0x1;
        /// The control is in a loop, can break and continue.
        const IN_LOOP = 0x2;
    }
}

struct BlockContext<'a> {
    flags: BlockFlags,
    expressions: &'a Arena<crate::Expression>,
    types: &'a Arena<crate::Type>,
    functions: &'a Arena<crate::Function>,
    return_type: Option<Handle<crate::Type>>,
}

impl<'a> BlockContext<'a> {
    fn with_flags(&self, flags: BlockFlags) -> Self {
        BlockContext {
            flags,
            expressions: self.expressions,
            types: self.types,
            functions: self.functions,
            return_type: self.return_type,
        }
    }

    fn get_expression(
        &self,
        handle: Handle<crate::Expression>,
    ) -> Result<&'a crate::Expression, FunctionError> {
        self.expressions
            .try_get(handle)
            .ok_or(FunctionError::InvalidExpression(handle))
    }
}

#[derive(Debug)]
pub struct Validator {
    //Note: this is a bit tricky: some of the front-ends as well as backends
    // already have to use the typifier, so the work here is redundant in a way.
    typifier: Typifier,
    type_flags: Vec<TypeFlags>,
    location_mask: BitSet,
    bind_group_masks: Vec<BitSet>,
    select_cases: FastHashSet<i32>,
    valid_expression_list: Vec<Handle<crate::Expression>>,
    valid_expression_set: BitSet,
}

#[derive(Clone, Debug, Error)]
pub enum TypeError {
    #[error("The {0:?} scalar width {1} is not supported")]
    InvalidWidth(crate::ScalarKind, crate::Bytes),
    #[error("The base handle {0:?} can not be resolved")]
    UnresolvedBase(Handle<crate::Type>),
    #[error("Expected data type, found {0:?}")]
    InvalidData(Handle<crate::Type>),
    #[error("Structure type {0:?} can not be a block structure")]
    InvalidBlockType(Handle<crate::Type>),
    #[error("Base type {0:?} for the array is invalid")]
    InvalidArrayBaseType(Handle<crate::Type>),
    #[error("The constant {0:?} can not be used for an array size")]
    InvalidArraySizeConstant(Handle<crate::Constant>),
    #[error("Field '{0}' can't be dynamically-sized, has type {1:?}")]
    InvalidDynamicArray(String, Handle<crate::Type>),
}

#[derive(Clone, Debug, Error)]
pub enum ConstantError {
    #[error("The type doesn't match the constant")]
    InvalidType,
    #[error("The component handle {0:?} can not be resolved")]
    UnresolvedComponent(Handle<crate::Constant>),
    #[error("The array size handle {0:?} can not be resolved")]
    UnresolvedSize(Handle<crate::Constant>),
}

#[derive(Clone, Debug, Error)]
pub enum GlobalVariableError {
    #[error("Usage isn't compatible with the storage class")]
    InvalidUsage,
    #[error("Type isn't compatible with the storage class")]
    InvalidType,
    #[error("Storage access {seen:?} exceeds the allowed {allowed:?}")]
    InvalidStorageAccess {
        allowed: crate::StorageAccess,
        seen: crate::StorageAccess,
    },
    #[error("Type flags {seen:?} do not meet the required {required:?}")]
    MissingTypeFlags {
        required: TypeFlags,
        seen: TypeFlags,
    },
    #[error("Binding decoration is missing or not applicable")]
    InvalidBinding,
}

#[derive(Clone, Debug, Error)]
pub enum LocalVariableError {
    #[error("Initializer doesn't match the variable type")]
    InitializerType,
}

#[derive(Clone, Debug, Error)]
pub enum VaryingError {
    #[error("The type does not match the varying")]
    InvalidType(Handle<crate::Type>),
    #[error("Interpolation is not valid")]
    InvalidInterpolation,
    #[error("BuiltIn {0:?} is not available at this stage")]
    InvalidBuiltInStage(crate::BuiltIn),
    #[error("BuiltIn type for {0:?} is invalid")]
    InvalidBuiltInType(crate::BuiltIn),
    #[error("Struct member {0} is missing a binding")]
    MemberMissingBinding(u32),
    #[error("Multiple bindings at location {location} are present")]
    BindingCollision { location: u32 },
}

#[derive(Clone, Debug, Error)]
pub enum ExpressionError {
    #[error("Is invalid")]
    Invalid,
    #[error("Used by a statement before it was introduced into the scope by any of the dominating blocks")]
    NotInScope,
}

#[derive(Clone, Debug, Error)]
pub enum CallError {
    #[error("Bad function")]
    Function,
    #[error("Argument {index} expression is invalid")]
    Argument {
        index: usize,
        #[source]
        error: ExpressionError,
    },
    #[error("Result expression {0:?} has already been introduced earlier")]
    ResultAlreadyInScope(Handle<crate::Expression>),
    #[error("Result value is invalid")]
    ResultValue(#[source] ExpressionError),
    #[error("Requires {required} arguments, but {seen} are provided")]
    ArgumentCount { required: usize, seen: usize },
    #[error("Argument {index} value {seen_expression:?} doesn't match the type {required:?}")]
    ArgumentType {
        index: usize,
        required: Handle<crate::Type>,
        seen_expression: Handle<crate::Expression>,
    },
    #[error("Result value {seen_expression:?} does not match the type {required:?}")]
    ResultType {
        required: Option<Handle<crate::Type>>,
        seen_expression: Option<Handle<crate::Expression>>,
    },
}

#[derive(Clone, Debug, Error)]
pub enum FunctionError {
    #[error(transparent)]
    Resolve(#[from] TypifyError),
    #[error("Expression {handle:?} is invalid")]
    Expression {
        handle: Handle<crate::Expression>,
        #[source]
        error: ExpressionError,
    },
    #[error("Expression {0:?} can't be introduced - it's already in scope")]
    ExpressionAlreadyInScope(Handle<crate::Expression>),
    #[error("Local variable {handle:?} '{name}' is invalid")]
    LocalVariable {
        handle: Handle<crate::LocalVariable>,
        name: String,
        #[source]
        error: LocalVariableError,
    },
    #[error("Argument '{name}' at index {index} has a type that can't be passed into functions.")]
    InvalidArgumentType { index: usize, name: String },
    #[error("There are instructions after `return`/`break`/`continue`")]
    InstructionsAfterReturn,
    #[error("The `break`/`continue` is used outside of a loop context")]
    BreakContinueOutsideOfLoop,
    #[error("The `return` is called within a `continuing` block")]
    InvalidReturnSpot,
    #[error("The `return` value {0:?} does not match the function return value")]
    InvalidReturnType(Option<Handle<crate::Expression>>),
    #[error("The `if` condition {0:?} is not a boolean scalar")]
    InvalidIfType(Handle<crate::Expression>),
    #[error("The `switch` value {0:?} is not an integer scalar")]
    InvalidSwitchType(Handle<crate::Expression>),
    #[error("Multiple `switch` cases for {0} are present")]
    ConflictingSwitchCase(i32),
    #[error("The pointer {0:?} doesn't relate to a valid destination for a store")]
    InvalidStorePointer(Handle<crate::Expression>),
    #[error("The value {0:?} can not be stored")]
    InvalidStoreValue(Handle<crate::Expression>),
    #[error("Store of {value:?} into {pointer:?} doesn't have matching types")]
    InvalidStoreTypes {
        pointer: Handle<crate::Expression>,
        value: Handle<crate::Expression>,
    },
    #[error("The image array can't be indexed by {0:?}")]
    InvalidArrayIndex(Handle<crate::Expression>),
    #[error("The expression {0:?} is currupted")]
    InvalidExpression(Handle<crate::Expression>),
    #[error("The expression {0:?} is not an image")]
    InvalidImage(Handle<crate::Expression>),
    #[error("Call to {function:?} is invalid")]
    InvalidCall {
        function: Handle<crate::Function>,
        #[source]
        error: CallError,
    },
}

#[derive(Clone, Debug, Error)]
pub enum EntryPointError {
    #[error("Multiple conflicting entry points")]
    Conflict,
    #[error("Early depth test is not applicable")]
    UnexpectedEarlyDepthTest,
    #[error("Workgroup size is not applicable")]
    UnexpectedWorkgroupSize,
    #[error("Workgroup size is out of range")]
    OutOfRangeWorkgroupSize,
    #[error("Global variable {0:?} is used incorrectly as {1:?}")]
    InvalidGlobalUsage(Handle<crate::GlobalVariable>, GlobalUse),
    #[error("Bindings for {0:?} conflict with other resource")]
    BindingCollision(Handle<crate::GlobalVariable>),
    #[error("Argument {0} varying error")]
    Argument(u32, #[source] VaryingError),
    #[error("Result varying error")]
    Result(#[source] VaryingError),
    #[error("Location {location} onterpolation of an integer has to be flat")]
    InvalidIntegerInterpolation { location: u32 },
    #[error(transparent)]
    Function(#[from] FunctionError),
}

#[derive(Clone, Debug, Error)]
pub enum ValidationError {
    #[error("Type {handle:?} '{name}' is invalid")]
    Type {
        handle: Handle<crate::Type>,
        name: String,
        #[source]
        error: TypeError,
    },
    #[error("Constant {handle:?} '{name}' is invalid")]
    Constant {
        handle: Handle<crate::Constant>,
        name: String,
        #[source]
        error: ConstantError,
    },
    #[error("Global variable {handle:?} '{name}' is invalid")]
    GlobalVariable {
        handle: Handle<crate::GlobalVariable>,
        name: String,
        #[source]
        error: GlobalVariableError,
    },
    #[error("Function {handle:?} '{name}' is invalid")]
    Function {
        handle: Handle<crate::Function>,
        name: String,
        #[source]
        error: FunctionError,
    },
    #[error("Entry point {name} at {stage:?} is invalid")]
    EntryPoint {
        stage: crate::ShaderStage,
        name: String,
        #[source]
        error: EntryPointError,
    },
    #[error(transparent)]
    Analysis(#[from] AnalysisError),
    #[error("Module is corrupted")]
    Corrupted,
}

fn storage_usage(access: crate::StorageAccess) -> GlobalUse {
    let mut storage_usage = GlobalUse::QUERY;
    if access.contains(crate::StorageAccess::LOAD) {
        storage_usage |= GlobalUse::READ;
    }
    if access.contains(crate::StorageAccess::STORE) {
        storage_usage |= GlobalUse::WRITE;
    }
    storage_usage
}

struct VaryingContext<'a> {
    ty: Handle<crate::Type>,
    stage: crate::ShaderStage,
    output: bool,
    types: &'a Arena<crate::Type>,
    location_mask: &'a mut BitSet,
}

impl VaryingContext<'_> {
    fn validate_impl(&mut self, binding: &crate::Binding) -> Result<(), VaryingError> {
        use crate::{
            BuiltIn as Bi, ScalarKind as Sk, ShaderStage as St, TypeInner as Ti, VectorSize as Vs,
        };

        let ty_inner = &self.types[self.ty].inner;
        match *binding {
            crate::Binding::BuiltIn(built_in) => {
                let width = 4;
                let (visible, type_good) = match built_in {
                    Bi::BaseInstance | Bi::BaseVertex | Bi::InstanceIndex | Bi::VertexIndex => (
                        self.stage == St::Vertex && !self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                    Bi::ClipDistance => (
                        self.stage == St::Vertex && self.output,
                        match *ty_inner {
                            Ti::Array { base, .. } => {
                                self.types[base].inner
                                    == Ti::Scalar {
                                        kind: Sk::Float,
                                        width,
                                    }
                            }
                            _ => false,
                        },
                    ),
                    Bi::PointSize => (
                        self.stage == St::Vertex && self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Float,
                                width,
                            },
                    ),
                    Bi::Position => (
                        self.stage == St::Vertex && self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Quad,
                                kind: Sk::Float,
                                width,
                            },
                    ),
                    Bi::FragCoord => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Quad,
                                kind: Sk::Float,
                                width,
                            },
                    ),
                    Bi::FragDepth => (
                        self.stage == St::Fragment && self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Float,
                                width,
                            },
                    ),
                    Bi::FrontFacing => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Bool,
                                width: crate::BOOL_WIDTH,
                            },
                    ),
                    Bi::SampleIndex | Bi::SampleMaskIn => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                    Bi::SampleMaskOut => (
                        self.stage == St::Fragment && self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                    Bi::LocalInvocationIndex => (
                        self.stage == St::Compute && !self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                    Bi::GlobalInvocationId
                    | Bi::LocalInvocationId
                    | Bi::WorkGroupId
                    | Bi::WorkGroupSize => (
                        self.stage == St::Compute && !self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Tri,
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                };

                if !visible {
                    return Err(VaryingError::InvalidBuiltInStage(built_in));
                }
                if !type_good {
                    log::warn!("Wrong builtin type: {:?}", ty_inner);
                    return Err(VaryingError::InvalidBuiltInType(built_in));
                }
            }
            crate::Binding::Location(location, interpolation) => {
                if !self.location_mask.insert(location as usize) {
                    return Err(VaryingError::BindingCollision { location });
                }
                let needs_interpolation =
                    self.stage == crate::ShaderStage::Fragment && !self.output;
                if !needs_interpolation && interpolation.is_some() {
                    return Err(VaryingError::InvalidInterpolation);
                }
                match ty_inner.scalar_kind() {
                    Some(crate::ScalarKind::Float) => {}
                    Some(_)
                        if needs_interpolation
                            && interpolation != Some(crate::Interpolation::Flat) =>
                    {
                        return Err(VaryingError::InvalidInterpolation);
                    }
                    _ => return Err(VaryingError::InvalidType(self.ty)),
                }
            }
        }

        Ok(())
    }

    fn validate(mut self, binding: Option<&crate::Binding>) -> Result<(), VaryingError> {
        match binding {
            Some(binding) => self.validate_impl(binding),
            None => {
                match self.types[self.ty].inner {
                    //TODO: check the member types
                    crate::TypeInner::Struct {
                        block: false,
                        ref members,
                    } => {
                        for (index, member) in members.iter().enumerate() {
                            self.ty = member.ty;
                            match member.binding {
                                None => {
                                    return Err(VaryingError::MemberMissingBinding(index as u32))
                                }
                                Some(ref binding) => self.validate_impl(binding)?,
                            }
                        }
                    }
                    _ => return Err(VaryingError::InvalidType(self.ty)),
                }
                Ok(())
            }
        }
    }
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new() -> Self {
        Validator {
            typifier: Typifier::new(),
            type_flags: Vec::new(),
            location_mask: BitSet::new(),
            bind_group_masks: Vec::new(),
            select_cases: FastHashSet::default(),
            valid_expression_list: Vec::new(),
            valid_expression_set: BitSet::new(),
        }
    }

    fn check_width(kind: crate::ScalarKind, width: crate::Bytes) -> bool {
        match kind {
            crate::ScalarKind::Bool => width == crate::BOOL_WIDTH,
            _ => width == 4,
        }
    }

    fn validate_type(
        &self,
        ty: &crate::Type,
        handle: Handle<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<TypeFlags, TypeError> {
        use crate::TypeInner as Ti;
        Ok(match ty.inner {
            Ti::Scalar { kind, width } | Ti::Vector { kind, width, .. } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::INTERFACE | TypeFlags::HOST_SHARED
            }
            Ti::Matrix { width, .. } => {
                if !Self::check_width(crate::ScalarKind::Float, width) {
                    return Err(TypeError::InvalidWidth(crate::ScalarKind::Float, width));
                }
                TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::INTERFACE | TypeFlags::HOST_SHARED
            }
            Ti::Pointer { base, class: _ } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                TypeFlags::DATA | TypeFlags::SIZED
            }
            Ti::ValuePointer {
                size: _,
                kind,
                width,
                class: _,
            } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeFlags::SIZED //TODO: `DATA`?
            }
            Ti::Array { base, size, stride } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                let base_flags = self.type_flags[base.index()];
                if !base_flags.contains(TypeFlags::DATA | TypeFlags::SIZED) {
                    return Err(TypeError::InvalidArrayBaseType(base));
                }

                let sized_flag = match size {
                    crate::ArraySize::Constant(const_handle) => {
                        match constants.try_get(const_handle) {
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Uint(_),
                                    },
                                ..
                            }) => {}
                            // Accept a signed integer size to avoid
                            // requiring an explicit uint
                            // literal. Type inference should make
                            // this unnecessary.
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Sint(_),
                                    },
                                ..
                            }) => {}
                            other => {
                                log::warn!("Array size {:?}", other);
                                return Err(TypeError::InvalidArraySizeConstant(const_handle));
                            }
                        }
                        TypeFlags::SIZED
                    }
                    crate::ArraySize::Dynamic => TypeFlags::empty(),
                };
                let base_mask = if stride.is_none() {
                    TypeFlags::INTERFACE
                } else {
                    TypeFlags::HOST_SHARED | TypeFlags::INTERFACE
                };
                TypeFlags::DATA | (base_flags & base_mask) | sized_flag
            }
            Ti::Struct { block, ref members } => {
                let mut flags = TypeFlags::all();
                for (i, member) in members.iter().enumerate() {
                    if member.ty >= handle {
                        return Err(TypeError::UnresolvedBase(member.ty));
                    }
                    let base_flags = self.type_flags[member.ty.index()];
                    flags &= base_flags;
                    if !base_flags.contains(TypeFlags::DATA) {
                        return Err(TypeError::InvalidData(member.ty));
                    }
                    if block && !base_flags.contains(TypeFlags::INTERFACE) {
                        return Err(TypeError::InvalidBlockType(member.ty));
                    }
                    // only the last field can be unsized
                    if i + 1 != members.len() && !base_flags.contains(TypeFlags::SIZED) {
                        let name = member.name.clone().unwrap_or_default();
                        return Err(TypeError::InvalidDynamicArray(name, member.ty));
                    }
                }
                //TODO: check the spans
                flags
            }
            Ti::Image { .. } | Ti::Sampler { .. } => TypeFlags::empty(),
        })
    }

    fn validate_constant(
        &self,
        handle: Handle<crate::Constant>,
        constants: &Arena<crate::Constant>,
        types: &Arena<crate::Type>,
    ) -> Result<(), ConstantError> {
        let con = &constants[handle];
        match con.inner {
            crate::ConstantInner::Scalar { width, ref value } => {
                if !Self::check_width(value.scalar_kind(), width) {
                    return Err(ConstantError::InvalidType);
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                match types[ty].inner {
                    crate::TypeInner::Array {
                        size: crate::ArraySize::Dynamic,
                        ..
                    } => {
                        return Err(ConstantError::InvalidType);
                    }
                    crate::TypeInner::Array {
                        size: crate::ArraySize::Constant(size_handle),
                        ..
                    } => {
                        if handle <= size_handle {
                            return Err(ConstantError::UnresolvedSize(size_handle));
                        }
                    }
                    _ => {} //TODO
                }
                if let Some(&comp) = components.iter().find(|&&comp| handle <= comp) {
                    return Err(ConstantError::UnresolvedComponent(comp));
                }
            }
        }
        Ok(())
    }

    fn validate_global_var(
        &self,
        var: &crate::GlobalVariable,
        types: &Arena<crate::Type>,
    ) -> Result<(), GlobalVariableError> {
        log::debug!("var {:?}", var);
        let (allowed_storage_access, required_type_flags, is_resource) = match var.class {
            crate::StorageClass::Function => return Err(GlobalVariableError::InvalidUsage),
            crate::StorageClass::Storage => {
                match types[var.ty].inner {
                    crate::TypeInner::Struct { .. } => (),
                    _ => return Err(GlobalVariableError::InvalidType),
                }
                (
                    crate::StorageAccess::all(),
                    TypeFlags::DATA | TypeFlags::HOST_SHARED,
                    true,
                )
            }
            crate::StorageClass::Uniform => {
                match types[var.ty].inner {
                    crate::TypeInner::Struct { .. } => (),
                    _ => return Err(GlobalVariableError::InvalidType),
                }
                (
                    crate::StorageAccess::empty(),
                    TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::HOST_SHARED,
                    true,
                )
            }
            crate::StorageClass::Handle => {
                let access = match types[var.ty].inner {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage(_),
                        ..
                    } => crate::StorageAccess::all(),
                    crate::TypeInner::Image { .. } | crate::TypeInner::Sampler { .. } => {
                        crate::StorageAccess::empty()
                    }
                    _ => return Err(GlobalVariableError::InvalidType),
                };
                (access, TypeFlags::empty(), true)
            }
            crate::StorageClass::Private | crate::StorageClass::WorkGroup => {
                (crate::StorageAccess::empty(), TypeFlags::DATA, false)
            }
            crate::StorageClass::PushConstant => (
                crate::StorageAccess::LOAD,
                TypeFlags::DATA | TypeFlags::HOST_SHARED,
                false,
            ),
        };

        if !allowed_storage_access.contains(var.storage_access) {
            return Err(GlobalVariableError::InvalidStorageAccess {
                seen: var.storage_access,
                allowed: allowed_storage_access,
            });
        }

        let type_flags = self.type_flags[var.ty.index()];
        if !type_flags.contains(required_type_flags) {
            return Err(GlobalVariableError::MissingTypeFlags {
                seen: type_flags,
                required: required_type_flags,
            });
        }

        if is_resource != var.binding.is_some() {
            return Err(GlobalVariableError::InvalidBinding);
        }

        Ok(())
    }

    fn validate_local_var(
        &self,
        var: &crate::LocalVariable,
        types: &Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<(), LocalVariableError> {
        log::debug!("var {:?}", var);
        if let Some(const_handle) = var.init {
            match constants[const_handle].inner {
                crate::ConstantInner::Scalar { width, ref value } => {
                    let ty_inner = crate::TypeInner::Scalar {
                        width,
                        kind: value.scalar_kind(),
                    };
                    if types[var.ty].inner != ty_inner {
                        return Err(LocalVariableError::InitializerType);
                    }
                }
                crate::ConstantInner::Composite { ty, components: _ } => {
                    if ty != var.ty {
                        return Err(LocalVariableError::InitializerType);
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_call(
        &mut self,
        function: Handle<crate::Function>,
        arguments: &[Handle<crate::Expression>],
        result: Option<Handle<crate::Expression>>,
        context: &BlockContext,
    ) -> Result<(), CallError> {
        let fun = context
            .functions
            .try_get(function)
            .ok_or(CallError::Function)?;
        if fun.arguments.len() != arguments.len() {
            return Err(CallError::ArgumentCount {
                required: fun.arguments.len(),
                seen: arguments.len(),
            });
        }
        for (index, (arg, &expr)) in fun.arguments.iter().zip(arguments).enumerate() {
            let ty = self
                .resolve_type_impl(expr, context.types)
                .map_err(|error| CallError::Argument { index, error })?;
            if ty != &context.types[arg.ty].inner {
                return Err(CallError::ArgumentType {
                    index,
                    required: arg.ty,
                    seen_expression: expr,
                });
            }
        }

        if let Some(expr) = result {
            if self.valid_expression_set.insert(expr.index()) {
                self.valid_expression_list.push(expr);
            } else {
                return Err(CallError::ResultAlreadyInScope(expr));
            }
        }

        let result_ty = result
            .map(|expr| self.resolve_type_impl(expr, context.types))
            .transpose()
            .map_err(CallError::ResultValue)?;
        let expected_ty = fun.result.as_ref().map(|fr| &context.types[fr.ty].inner);
        if result_ty != expected_ty {
            log::error!(
                "Called function returns {:?} where {:?} is expected",
                result_ty,
                expected_ty
            );
            return Err(CallError::ResultType {
                required: fun.result.as_ref().map(|fr| fr.ty),
                seen_expression: result,
            });
        }
        Ok(())
    }

    fn resolve_type_impl<'a>(
        &'a self,
        handle: Handle<crate::Expression>,
        types: &'a Arena<crate::Type>,
    ) -> Result<&'a crate::TypeInner, ExpressionError> {
        if !self.valid_expression_set.contains(handle.index()) {
            return Err(ExpressionError::NotInScope);
        }
        self.typifier
            .try_get(handle, types)
            .ok_or(ExpressionError::Invalid)
    }

    fn resolve_type<'a>(
        &'a self,
        handle: Handle<crate::Expression>,
        types: &'a Arena<crate::Type>,
    ) -> Result<&'a crate::TypeInner, FunctionError> {
        self.resolve_type_impl(handle, types)
            .map_err(|error| FunctionError::Expression { handle, error })
    }

    fn validate_block_impl(
        &mut self,
        statements: &[crate::Statement],
        context: &BlockContext,
    ) -> Result<(), FunctionError> {
        use crate::{Statement as S, TypeInner as Ti};
        let mut finished = false;
        for statement in statements {
            if finished {
                return Err(FunctionError::InstructionsAfterReturn);
            }
            match *statement {
                S::Emit(ref range) => {
                    for handle in range.clone() {
                        if self.valid_expression_set.insert(handle.index()) {
                            self.valid_expression_list.push(handle);
                        } else {
                            return Err(FunctionError::ExpressionAlreadyInScope(handle));
                        }
                    }
                }
                S::Block(ref block) => self.validate_block(block, context)?,
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    match *self.resolve_type(condition, context.types)? {
                        Ti::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: _,
                        } => {}
                        _ => return Err(FunctionError::InvalidIfType(condition)),
                    }
                    self.validate_block(accept, context)?;
                    self.validate_block(reject, context)?;
                }
                S::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    match *self.resolve_type(selector, context.types)? {
                        Ti::Scalar {
                            kind: crate::ScalarKind::Sint,
                            width: _,
                        } => {}
                        _ => return Err(FunctionError::InvalidSwitchType(selector)),
                    }
                    self.select_cases.clear();
                    for case in cases {
                        if !self.select_cases.insert(case.value) {
                            return Err(FunctionError::ConflictingSwitchCase(case.value));
                        }
                    }
                    for case in cases {
                        self.validate_block(&case.body, context)?;
                    }
                    self.validate_block(default, context)?;
                }
                S::Loop {
                    ref body,
                    ref continuing,
                } => {
                    // special handling for block scoping is needed here,
                    // because the continuing{} block inherits the scope
                    let base_expression_count = self.valid_expression_list.len();
                    self.validate_block_impl(
                        body,
                        &context.with_flags(BlockFlags::CAN_JUMP | BlockFlags::IN_LOOP),
                    )?;
                    self.validate_block_impl(continuing, &context.with_flags(BlockFlags::empty()))?;
                    for handle in self.valid_expression_list.drain(base_expression_count..) {
                        self.valid_expression_set.remove(handle.index());
                    }
                }
                S::Break | S::Continue => {
                    if !context.flags.contains(BlockFlags::IN_LOOP) {
                        return Err(FunctionError::BreakContinueOutsideOfLoop);
                    }
                    finished = true;
                }
                S::Return { value } => {
                    if !context.flags.contains(BlockFlags::CAN_JUMP) {
                        return Err(FunctionError::InvalidReturnSpot);
                    }
                    let value_ty = value
                        .map(|expr| self.resolve_type(expr, context.types))
                        .transpose()?;
                    let expected_ty = context.return_type.map(|ty| &context.types[ty].inner);
                    if value_ty != expected_ty {
                        log::error!(
                            "Returning {:?} where {:?} is expected",
                            value_ty,
                            expected_ty
                        );
                        return Err(FunctionError::InvalidReturnType(value));
                    }
                    finished = true;
                }
                S::Kill => {
                    finished = true;
                }
                S::Store { pointer, value } => {
                    let mut current = pointer;
                    loop {
                        self.typifier.try_get(current, context.types).ok_or(
                            FunctionError::Expression {
                                handle: current,
                                error: ExpressionError::Invalid,
                            },
                        )?;
                        match context.expressions[current] {
                            crate::Expression::Access { base, .. }
                            | crate::Expression::AccessIndex { base, .. } => current = base,
                            crate::Expression::LocalVariable(_)
                            | crate::Expression::GlobalVariable(_) => break,
                            _ => return Err(FunctionError::InvalidStorePointer(current)),
                        }
                    }

                    let value_ty = self.resolve_type(value, context.types)?;
                    match *value_ty {
                        Ti::Image { .. } | Ti::Sampler { .. } => {
                            return Err(FunctionError::InvalidStoreValue(value));
                        }
                        _ => {}
                    }
                    let good = match self.typifier.try_get(pointer, context.types) {
                        Some(&Ti::Pointer { base, class: _ }) => {
                            *value_ty == context.types[base].inner
                        }
                        Some(&Ti::ValuePointer {
                            size: Some(size),
                            kind,
                            width,
                            class: _,
                        }) => *value_ty == Ti::Vector { size, kind, width },
                        Some(&Ti::ValuePointer {
                            size: None,
                            kind,
                            width,
                            class: _,
                        }) => *value_ty == Ti::Scalar { kind, width },
                        _ => false,
                    };
                    if !good {
                        return Err(FunctionError::InvalidStoreTypes { pointer, value });
                    }
                }
                S::ImageStore {
                    image,
                    coordinate: _,
                    array_index,
                    value,
                } => {
                    let _expected_coordinate_ty = match *context.get_expression(image)? {
                        crate::Expression::GlobalVariable(_var_handle) => (), //TODO
                        _ => return Err(FunctionError::InvalidImage(image)),
                    };
                    let value_ty = self.typifier.get(value, context.types);
                    match *value_ty {
                        Ti::Scalar { .. } | Ti::Vector { .. } => {}
                        _ => {
                            return Err(FunctionError::InvalidStoreValue(value));
                        }
                    }
                    if let Some(expr) = array_index {
                        match *self.typifier.get(expr, context.types) {
                            Ti::Scalar {
                                kind: crate::ScalarKind::Sint,
                                width: _,
                            } => (),
                            _ => return Err(FunctionError::InvalidArrayIndex(expr)),
                        }
                    }
                }
                S::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    if let Err(error) = self.validate_call(function, arguments, result, context) {
                        return Err(FunctionError::InvalidCall { function, error });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_block(
        &mut self,
        statements: &[crate::Statement],
        context: &BlockContext,
    ) -> Result<(), FunctionError> {
        let base_expression_count = self.valid_expression_list.len();
        self.validate_block_impl(statements, context)?;
        for handle in self.valid_expression_list.drain(base_expression_count..) {
            self.valid_expression_set.remove(handle.index());
        }
        Ok(())
    }

    fn validate_function(
        &mut self,
        fun: &crate::Function,
        _info: &FunctionInfo,
        module: &crate::Module,
    ) -> Result<(), FunctionError> {
        let resolve_ctx = ResolveContext {
            constants: &module.constants,
            global_vars: &module.global_variables,
            local_vars: &fun.local_variables,
            functions: &module.functions,
            arguments: &fun.arguments,
        };
        self.typifier
            .resolve_all(&fun.expressions, &module.types, &resolve_ctx)?;

        for (var_handle, var) in fun.local_variables.iter() {
            self.validate_local_var(var, &module.types, &module.constants)
                .map_err(|error| FunctionError::LocalVariable {
                    handle: var_handle,
                    name: var.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        for (index, argument) in fun.arguments.iter().enumerate() {
            if !self.type_flags[argument.ty.index()].contains(TypeFlags::DATA) {
                return Err(FunctionError::InvalidArgumentType {
                    index,
                    name: argument.name.clone().unwrap_or_default(),
                });
            }
        }

        self.valid_expression_set.clear();
        for (handle, expr) in fun.expressions.iter() {
            if expr.needs_pre_emit() {
                self.valid_expression_set.insert(handle.index());
            }
        }

        self.validate_block(
            &fun.body,
            &BlockContext {
                flags: BlockFlags::CAN_JUMP,
                expressions: &fun.expressions,
                types: &module.types,
                functions: &module.functions,
                return_type: fun.result.as_ref().map(|fr| fr.ty),
            },
        )
    }

    fn validate_entry_point(
        &mut self,
        ep: &crate::EntryPoint,
        info: &FunctionInfo,
        module: &crate::Module,
    ) -> Result<(), EntryPointError> {
        if ep.early_depth_test.is_some() && ep.stage != crate::ShaderStage::Fragment {
            return Err(EntryPointError::UnexpectedEarlyDepthTest);
        }
        if ep.stage == crate::ShaderStage::Compute {
            if ep
                .workgroup_size
                .iter()
                .any(|&s| s == 0 || s > MAX_WORKGROUP_SIZE)
            {
                return Err(EntryPointError::OutOfRangeWorkgroupSize);
            }
        } else if ep.workgroup_size != [0; 3] {
            return Err(EntryPointError::UnexpectedWorkgroupSize);
        }

        self.location_mask.clear();
        for (index, fa) in ep.function.arguments.iter().enumerate() {
            let ctx = VaryingContext {
                ty: fa.ty,
                stage: ep.stage,
                output: false,
                types: &module.types,
                location_mask: &mut self.location_mask,
            };
            ctx.validate(fa.binding.as_ref())
                .map_err(|e| EntryPointError::Argument(index as u32, e))?;
        }

        self.location_mask.clear();
        if let Some(ref fr) = ep.function.result {
            let ctx = VaryingContext {
                ty: fr.ty,
                stage: ep.stage,
                output: true,
                types: &module.types,
                location_mask: &mut self.location_mask,
            };
            ctx.validate(fr.binding.as_ref())
                .map_err(EntryPointError::Result)?;
        }

        for bg in self.bind_group_masks.iter_mut() {
            bg.clear();
        }
        for (var_handle, var) in module.global_variables.iter() {
            let usage = info[var_handle];
            if usage.is_empty() {
                continue;
            }

            let allowed_usage = match var.class {
                crate::StorageClass::Function => unreachable!(),
                crate::StorageClass::Uniform => GlobalUse::READ | GlobalUse::QUERY,
                crate::StorageClass::Storage => storage_usage(var.storage_access),
                crate::StorageClass::Handle => match module.types[var.ty].inner {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage(_),
                        ..
                    } => storage_usage(var.storage_access),
                    _ => GlobalUse::READ | GlobalUse::QUERY,
                },
                crate::StorageClass::Private | crate::StorageClass::WorkGroup => GlobalUse::all(),
                crate::StorageClass::PushConstant => GlobalUse::READ,
            };
            if !allowed_usage.contains(usage) {
                log::warn!("\tUsage error for: {:?}", var);
                log::warn!(
                    "\tAllowed usage: {:?}, requested: {:?}",
                    allowed_usage,
                    usage
                );
                return Err(EntryPointError::InvalidGlobalUsage(var_handle, usage));
            }

            if let Some(ref bind) = var.binding {
                while self.bind_group_masks.len() <= bind.group as usize {
                    self.bind_group_masks.push(BitSet::new());
                }
                if !self.bind_group_masks[bind.group as usize].insert(bind.binding as usize) {
                    return Err(EntryPointError::BindingCollision(var_handle));
                }
            }
        }

        self.validate_function(&ep.function, info, module)?;
        Ok(())
    }

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<Analysis, ValidationError> {
        self.typifier.clear();
        self.type_flags.clear();
        self.type_flags
            .resize(module.types.len(), TypeFlags::empty());

        let analysis = Analysis::new(module)?;

        for (handle, constant) in module.constants.iter() {
            self.validate_constant(handle, &module.constants, &module.types)
                .map_err(|error| ValidationError::Constant {
                    handle,
                    name: constant.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        // doing after the globals, so that `type_flags` is ready
        for (handle, ty) in module.types.iter() {
            let ty_flags = self
                .validate_type(ty, handle, &module.constants)
                .map_err(|error| ValidationError::Type {
                    handle,
                    name: ty.name.clone().unwrap_or_default(),
                    error,
                })?;
            self.type_flags[handle.index()] = ty_flags;
        }

        for (var_handle, var) in module.global_variables.iter() {
            self.validate_global_var(var, &module.types)
                .map_err(|error| ValidationError::GlobalVariable {
                    handle: var_handle,
                    name: var.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        for (handle, fun) in module.functions.iter() {
            self.validate_function(fun, &analysis[handle], module)
                .map_err(|error| ValidationError::Function {
                    handle,
                    name: fun.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        let mut ep_map = FastHashSet::default();
        for (index, ep) in module.entry_points.iter().enumerate() {
            if !ep_map.insert((ep.stage, &ep.name)) {
                return Err(ValidationError::EntryPoint {
                    stage: ep.stage,
                    name: ep.name.clone(),
                    error: EntryPointError::Conflict,
                });
            }
            let info = analysis.get_entry_point(index);
            self.validate_entry_point(ep, info, module)
                .map_err(|error| ValidationError::EntryPoint {
                    stage: ep.stage,
                    name: ep.name.clone(),
                    error,
                })?;
        }

        Ok(analysis)
    }
}
