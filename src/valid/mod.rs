mod analyzer;
mod compose;
mod expression;
mod function;
mod interface;
mod r#type;

use crate::{
    arena::{Arena, Handle},
    FastHashSet,
};
use bit_set::BitSet;
use std::ops;

//TODO: analyze the model at the same time as we validate it,
// merge the corresponding matches over expressions and statements.

pub use analyzer::{ExpressionInfo, FunctionInfo, GlobalUse, Uniformity, UniformityRequirements};
pub use compose::ComposeError;
pub use expression::ExpressionError;
pub use function::{CallError, FunctionError, LocalVariableError};
pub use interface::{EntryPointError, GlobalVariableError, VaryingError};
pub use r#type::{Disalignment, TypeError, TypeFlags};

bitflags::bitflags! {
    /// Validation flags.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct ValidationFlags: u8 {
        /// Expressions.
        const EXPRESSIONS = 0x1;
        /// Statements and blocks of them.
        const BLOCKS = 0x2;
        /// Uniformity of control flow for operations that require it.
        const CONTROL_FLOW_UNIFORMITY = 0x4;
        /// Host-shareable structure layouts.
        const STRUCT_LAYOUTS = 0x8;
        /// Constants.
        const CONSTANTS = 0x10;
    }
}

impl Default for ValidationFlags {
    fn default() -> Self {
        Self::all()
    }
}

#[must_use]
bitflags::bitflags! {
    /// Allowed IR capabilities.
    #[derive(Default)]
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct Capabilities: u8 {
        /// Support for `StorageClass:PushConstant`.
        const PUSH_CONSTANT = 0x1;
        /// Float values with width = 8.
        const FLOAT64 = 0x2;
    }
}

bitflags::bitflags! {
    /// Validation flags.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct ShaderStages: u8 {
        const VERTEX = 0x1;
        const FRAGMENT = 0x2;
        const COMPUTE = 0x4;
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct ModuleInfo {
    functions: Vec<FunctionInfo>,
    entry_points: Vec<FunctionInfo>,
}

impl ops::Index<Handle<crate::Function>> for ModuleInfo {
    type Output = FunctionInfo;
    fn index(&self, handle: Handle<crate::Function>) -> &Self::Output {
        &self.functions[handle.index()]
    }
}

#[derive(Debug)]
pub struct Validator {
    flags: ValidationFlags,
    capabilities: Capabilities,
    types: Vec<r#type::TypeInfo>,
    location_mask: BitSet,
    bind_group_masks: Vec<BitSet>,
    select_cases: FastHashSet<i32>,
    valid_expression_list: Vec<Handle<crate::Expression>>,
    valid_expression_set: BitSet,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ConstantError {
    #[error("The type doesn't match the constant")]
    InvalidType,
    #[error("The component handle {0:?} can not be resolved")]
    UnresolvedComponent(Handle<crate::Constant>),
    #[error("The array size handle {0:?} can not be resolved")]
    UnresolvedSize(Handle<crate::Constant>),
    #[error(transparent)]
    Compose(#[from] ComposeError),
}

#[derive(Clone, Debug, thiserror::Error)]
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

    fn image_storage_coordinates(&self) -> Option<crate::ImageDimension> {
        match *self {
            Self::Scalar {
                kind: crate::ScalarKind::Sint,
                ..
            } => Some(crate::ImageDimension::D1),
            Self::Vector {
                size: crate::VectorSize::Bi,
                kind: crate::ScalarKind::Sint,
                ..
            } => Some(crate::ImageDimension::D2),
            Self::Vector {
                size: crate::VectorSize::Tri,
                kind: crate::ScalarKind::Sint,
                ..
            } => Some(crate::ImageDimension::D3),
            _ => None,
        }
    }
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new(flags: ValidationFlags, capabilities: Capabilities) -> Self {
        Validator {
            flags,
            capabilities,
            types: Vec::new(),
            location_mask: BitSet::new(),
            bind_group_masks: Vec::new(),
            select_cases: FastHashSet::default(),
            valid_expression_list: Vec::new(),
            valid_expression_set: BitSet::new(),
        }
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
                if !self.check_width(value.scalar_kind(), width) {
                    return Err(ConstantError::InvalidType);
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                match types[ty].inner {
                    crate::TypeInner::Array {
                        size: crate::ArraySize::Constant(size_handle),
                        ..
                    } if handle <= size_handle => {
                        return Err(ConstantError::UnresolvedSize(size_handle));
                    }
                    _ => {}
                }
                if let Some(&comp) = components.iter().find(|&&comp| handle <= comp) {
                    return Err(ConstantError::UnresolvedComponent(comp));
                }
                compose::validate_compose(
                    ty,
                    constants,
                    types,
                    components
                        .iter()
                        .map(|&component| constants[component].inner.resolve_type()),
                )?;
            }
        }
        Ok(())
    }

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<ModuleInfo, ValidationError> {
        self.reset_types(module.types.len());

        if self.flags.contains(ValidationFlags::CONSTANTS) {
            for (handle, constant) in module.constants.iter() {
                self.validate_constant(handle, &module.constants, &module.types)
                    .map_err(|error| ValidationError::Constant {
                        handle,
                        name: constant.name.clone().unwrap_or_default(),
                        error,
                    })?;
            }
        }

        // doing after the globals, so that `type_flags` is ready
        for (handle, ty) in module.types.iter() {
            let ty_info = self
                .validate_type(handle, &module.types, &module.constants)
                .map_err(|error| ValidationError::Type {
                    handle,
                    name: ty.name.clone().unwrap_or_default(),
                    error,
                })?;
            self.types[handle.index()] = ty_info;
        }

        for (var_handle, var) in module.global_variables.iter() {
            self.validate_global_var(var, &module.types)
                .map_err(|error| ValidationError::GlobalVariable {
                    handle: var_handle,
                    name: var.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        let mut mod_info = ModuleInfo {
            functions: Vec::with_capacity(module.functions.len()),
            entry_points: Vec::with_capacity(module.entry_points.len()),
        };

        for (handle, fun) in module.functions.iter() {
            match self.validate_function(fun, module, &mod_info) {
                Ok(info) => mod_info.functions.push(info),
                Err(error) => {
                    return Err(ValidationError::Function {
                        handle,
                        name: fun.name.clone().unwrap_or_default(),
                        error,
                    })
                }
            }
        }

        let mut ep_map = FastHashSet::default();
        for ep in module.entry_points.iter() {
            if !ep_map.insert((ep.stage, &ep.name)) {
                return Err(ValidationError::EntryPoint {
                    stage: ep.stage,
                    name: ep.name.clone(),
                    error: EntryPointError::Conflict,
                });
            }

            match self.validate_entry_point(ep, module, &mod_info) {
                Ok(info) => mod_info.entry_points.push(info),
                Err(error) => {
                    return Err(ValidationError::EntryPoint {
                        stage: ep.stage,
                        name: ep.name.clone(),
                        error,
                    })
                }
            }
        }

        Ok(mod_info)
    }
}
