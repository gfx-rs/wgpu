mod analyzer;
mod expression;
mod function;
mod interface;

use crate::{
    arena::{Arena, Handle},
    proc::Typifier,
    FastHashSet,
};
use bit_set::BitSet;
use thiserror::Error;

//TODO: analyze the model at the same time as we validate it,
// merge the corresponding matches over expressions and statements.
pub use analyzer::{
    AnalysisError, AnalysisFlags, ExpressionInfo, FunctionInfo, GlobalUse, ModuleInfo, Uniformity,
    UniformityRequirements,
};
pub use expression::ExpressionError;
pub use function::{CallError, FunctionError, LocalVariableError};
pub use interface::{EntryPointError, GlobalVariableError, VaryingError};

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

#[derive(Debug)]
pub struct Validator {
    analysis_flags: AnalysisFlags,
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

impl crate::TypeInner {
    fn is_sized(&self) -> bool {
        match *self {
            Self::Scalar { .. }
            | Self::Vector { .. }
            | Self::Matrix { .. }
            | Self::Array {
                size: crate::ArraySize::Constant(_),
                ..
            }
            | Self::Pointer { .. }
            | Self::ValuePointer { .. }
            | Self::Struct { .. } => true,
            Self::Array { .. } | Self::Image { .. } | Self::Sampler { .. } => false,
        }
    }
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new(analysis_flags: AnalysisFlags) -> Self {
        Validator {
            analysis_flags,
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
            Ti::Array {
                base,
                size,
                stride: _,
            } => {
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
                let base_mask = TypeFlags::HOST_SHARED | TypeFlags::INTERFACE;
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

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<ModuleInfo, ValidationError> {
        self.typifier.clear();
        self.type_flags.clear();
        self.type_flags
            .resize(module.types.len(), TypeFlags::empty());

        let analysis = ModuleInfo::new(module, self.analysis_flags)?;

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
