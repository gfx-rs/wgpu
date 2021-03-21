mod analyzer;
mod expression;
mod function;
mod interface;

use crate::{
    arena::{Arena, Handle},
    proc::{Layouter, Typifier},
    FastHashSet,
};
use bit_set::BitSet;
use thiserror::Error;

//TODO: analyze the model at the same time as we validate it,
// merge the corresponding matches over expressions and statements.
pub use analyzer::{
    AnalysisError, ExpressionInfo, FunctionInfo, GlobalUse, ModuleInfo, Uniformity,
    UniformityRequirements,
};
pub use expression::ExpressionError;
pub use function::{CallError, FunctionError, LocalVariableError};
pub use interface::{EntryPointError, GlobalVariableError, VaryingError};

bitflags::bitflags! {
    /// Validation flags.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct ValidationFlags: u8 {
        const EXPRESSIONS = 0x1;
        const BLOCKS = 0x2;
        const CONTROL_FLOW_UNIFORMITY = 0x4;
    }
}

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

#[derive(Clone, Debug, Error)]
pub enum Disalignment {
    #[error("The array stride {stride} is not a multiple of the required alignment {alignment}")]
    ArrayStride { stride: u32, alignment: u32 },
    #[error("The struct size {size}, is not a multiple of the required alignment {alignment}")]
    StructSize { size: u32, alignment: u32 },
    #[error("The struct member[{index}] offset {offset} is not a multiple of the required alignment {alignment}")]
    Member {
        index: u32,
        offset: u32,
        alignment: u32,
    },
    #[error("The struct member[{index}] is not statically sized")]
    UnsizedMember { index: u32 },
}

// Only makes sense if `flags.contains(HOST_SHARED)`
type LayoutCompatibility = Result<(), (Handle<crate::Type>, Disalignment)>;

// For the uniform buffer alignment, array strides and struct sizes must be multiples of 16.
const UNIFORM_LAYOUT_ALIGNMENT_MASK: u32 = 0xF;

#[derive(Clone, Debug)]
struct TypeInfo {
    flags: TypeFlags,
    uniform_layout: LayoutCompatibility,
    storage_layout: LayoutCompatibility,
}

impl TypeInfo {
    fn new() -> Self {
        TypeInfo {
            flags: TypeFlags::empty(),
            uniform_layout: Ok(()),
            storage_layout: Ok(()),
        }
    }

    fn from_flags(flags: TypeFlags) -> Self {
        TypeInfo {
            flags,
            uniform_layout: Ok(()),
            storage_layout: Ok(()),
        }
    }
}

#[derive(Debug)]
pub struct Validator {
    flags: ValidationFlags,
    //Note: this is a bit tricky: some of the front-ends as well as backends
    // already have to use the typifier, so the work here is redundant in a way.
    typifier: Typifier,
    types: Vec<TypeInfo>,
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
    #[error(
        "Array stride {stride} is not a multiple of the base element alignment {base_alignment}"
    )]
    UnalignedArrayStride { stride: u32, base_alignment: u32 },
    #[error("Array stride {stride} is smaller than the base element size {base_size}")]
    InsufficientArrayStride { stride: u32, base_size: u32 },
    #[error("Field '{0}' can't be dynamically-sized, has type {1:?}")]
    InvalidDynamicArray(String, Handle<crate::Type>),
    #[error("Structure member[{index}] size {size} is not a sufficient to hold {base_size}")]
    InsufficientMemberSize {
        index: u32,
        size: u32,
        base_size: u32,
    },
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
    pub fn new(flags: ValidationFlags) -> Self {
        Validator {
            flags,
            typifier: Typifier::new(),
            types: Vec::new(),
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
        layouter: &Layouter,
    ) -> Result<TypeInfo, TypeError> {
        use crate::TypeInner as Ti;
        Ok(match ty.inner {
            Ti::Scalar { kind, width } | Ti::Vector { kind, width, .. } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeInfo::from_flags(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                )
            }
            Ti::Matrix { width, .. } => {
                if !Self::check_width(crate::ScalarKind::Float, width) {
                    return Err(TypeError::InvalidWidth(crate::ScalarKind::Float, width));
                }
                TypeInfo::from_flags(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                )
            }
            Ti::Pointer { base, class: _ } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                TypeInfo::from_flags(TypeFlags::DATA | TypeFlags::SIZED)
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
                TypeInfo::from_flags(TypeFlags::SIZED)
            }
            Ti::Array { base, size, stride } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA | TypeFlags::SIZED) {
                    return Err(TypeError::InvalidArrayBaseType(base));
                }

                let base_layout = &layouter[base];
                if let Some(stride) = stride {
                    if stride.get() % base_layout.alignment.get() != 0 {
                        return Err(TypeError::UnalignedArrayStride {
                            stride: stride.get(),
                            base_alignment: base_layout.alignment.get(),
                        });
                    }
                    if stride.get() < base_layout.size {
                        return Err(TypeError::InsufficientArrayStride {
                            stride: stride.get(),
                            base_size: base_layout.size,
                        });
                    }
                }

                let (sized_flag, uniform_layout) = match size {
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

                        let effective_alignment = match stride {
                            Some(stride) => stride.get(),
                            None => base_layout.size,
                        };
                        let uniform_layout =
                            if effective_alignment & UNIFORM_LAYOUT_ALIGNMENT_MASK == 0 {
                                base_info.uniform_layout.clone()
                            } else {
                                Err((
                                    handle,
                                    Disalignment::ArrayStride {
                                        stride: effective_alignment,
                                        alignment: effective_alignment,
                                    },
                                ))
                            };
                        (TypeFlags::SIZED, uniform_layout)
                    }
                    //Note: this will be detected at the struct level
                    crate::ArraySize::Dynamic => (TypeFlags::empty(), Ok(())),
                };

                let base_mask = TypeFlags::HOST_SHARED | TypeFlags::INTERFACE;
                TypeInfo {
                    flags: TypeFlags::DATA | (base_info.flags & base_mask) | sized_flag,
                    uniform_layout,
                    storage_layout: base_info.storage_layout.clone(),
                }
            }
            Ti::Struct { block, ref members } => {
                let mut flags = TypeFlags::all();
                let mut uniform_layout = Ok(());
                let mut storage_layout = Ok(());
                let mut offset = 0;
                for (i, member) in members.iter().enumerate() {
                    if member.ty >= handle {
                        return Err(TypeError::UnresolvedBase(member.ty));
                    }
                    let base_info = &self.types[member.ty.index()];
                    flags &= base_info.flags;
                    if !base_info.flags.contains(TypeFlags::DATA) {
                        return Err(TypeError::InvalidData(member.ty));
                    }
                    if block && !base_info.flags.contains(TypeFlags::INTERFACE) {
                        return Err(TypeError::InvalidBlockType(member.ty));
                    }

                    let base_layout = &layouter[member.ty];
                    let (range, _alignment) = layouter.member_placement(offset, member);
                    if range.end - range.start < base_layout.size {
                        return Err(TypeError::InsufficientMemberSize {
                            index: i as u32,
                            size: range.end - range.start,
                            base_size: base_layout.size,
                        });
                    }
                    if range.start % base_layout.alignment.get() != 0 {
                        let result = Err((
                            handle,
                            Disalignment::Member {
                                index: i as u32,
                                offset: range.start,
                                alignment: base_layout.alignment.get(),
                            },
                        ));
                        uniform_layout = uniform_layout.or_else(|_| result.clone());
                        storage_layout = storage_layout.or(result);
                    }
                    offset = range.end;

                    // only the last field can be unsized
                    if !base_info.flags.contains(TypeFlags::SIZED) {
                        if i + 1 != members.len() {
                            let name = member.name.clone().unwrap_or_default();
                            return Err(TypeError::InvalidDynamicArray(name, member.ty));
                        }
                        if uniform_layout.is_ok() {
                            uniform_layout =
                                Err((handle, Disalignment::UnsizedMember { index: i as u32 }));
                        }
                    }

                    uniform_layout = uniform_layout.or_else(|_| base_info.uniform_layout.clone());
                    storage_layout = storage_layout.or_else(|_| base_info.storage_layout.clone());
                }

                if uniform_layout.is_ok() && offset % UNIFORM_LAYOUT_ALIGNMENT_MASK == 0 {
                    uniform_layout = Err((
                        handle,
                        Disalignment::StructSize {
                            size: offset,
                            alignment: UNIFORM_LAYOUT_ALIGNMENT_MASK,
                        },
                    ));
                }
                TypeInfo {
                    flags,
                    uniform_layout,
                    storage_layout,
                }
            }
            Ti::Image { .. } | Ti::Sampler { .. } => TypeInfo::from_flags(TypeFlags::empty()),
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
        self.types.clear();
        self.types.resize(module.types.len(), TypeInfo::new());

        let mod_info = ModuleInfo::new(module, self.flags)?;

        let layouter = Layouter::new(&module.types, &module.constants);

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
            let ty_info = self
                .validate_type(ty, handle, &module.constants, &layouter)
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

        for (handle, fun) in module.functions.iter() {
            self.validate_function(fun, &mod_info[handle], module)
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
            let info = mod_info.get_entry_point(index);
            self.validate_entry_point(ep, info, module)
                .map_err(|error| ValidationError::EntryPoint {
                    stage: ep.stage,
                    name: ep.name.clone(),
                    error,
                })?;
        }

        Ok(mod_info)
    }
}
