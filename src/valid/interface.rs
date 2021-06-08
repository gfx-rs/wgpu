use super::{
    analyzer::{FunctionInfo, GlobalUse},
    Capabilities, Disalignment, FunctionError, ModuleInfo, ShaderStages, TypeFlags,
    ValidationFlags,
};
use crate::arena::{Arena, Handle};

use bit_set::BitSet;

const MAX_WORKGROUP_SIZE: u32 = 0x4000;

#[derive(Clone, Debug, thiserror::Error)]
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
    #[error("Capability {0:?} is not supported")]
    UnsupportedCapability(Capabilities),
    #[error("Binding decoration is missing or not applicable")]
    InvalidBinding,
    #[error("Alignment requirements for this storage class are not met by {0:?}")]
    Alignment(Handle<crate::Type>, #[source] Disalignment),
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum VaryingError {
    #[error("The type {0:?} does not match the varying")]
    InvalidType(Handle<crate::Type>),
    #[error("Interpolation is not valid")]
    InvalidInterpolation,
    #[error("Interpolation must be specified on vertex shader outputs and fragment shader inputs")]
    MissingInterpolation,
    #[error("Built-in {0:?} is not available at this stage")]
    InvalidBuiltInStage(crate::BuiltIn),
    #[error("Built-in type for {0:?} is invalid")]
    InvalidBuiltInType(crate::BuiltIn),
    #[error("Entry point arguments and return values must all have bindings")]
    MissingBinding,
    #[error("Struct member {0} is missing a binding")]
    MemberMissingBinding(u32),
    #[error("Multiple bindings at location {location} are present")]
    BindingCollision { location: u32 },
    #[error("Built-in {0:?} is present more than once")]
    DuplicateBuiltIn(crate::BuiltIn),
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum EntryPointError {
    #[error("Multiple conflicting entry points")]
    Conflict,
    #[error("Early depth test is not applicable")]
    UnexpectedEarlyDepthTest,
    #[error("Workgroup size is not applicable")]
    UnexpectedWorkgroupSize,
    #[error("Workgroup size is out of range")]
    OutOfRangeWorkgroupSize,
    #[error("Uses operations forbidden at this stage")]
    ForbiddenStageOperations,
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
    built_in_mask: u32,
}

impl VaryingContext<'_> {
    fn validate_impl(&mut self, binding: &crate::Binding) -> Result<(), VaryingError> {
        use crate::{
            BuiltIn as Bi, ScalarKind as Sk, ShaderStage as St, TypeInner as Ti, VectorSize as Vs,
        };

        let ty_inner = &self.types[self.ty].inner;
        match *binding {
            crate::Binding::BuiltIn(built_in) => {
                let bit = 1 << built_in as u32;
                if self.built_in_mask & bit != 0 {
                    return Err(VaryingError::DuplicateBuiltIn(built_in));
                }
                self.built_in_mask |= bit;

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
                    Bi::ClipDistance | Bi::CullDistance => (
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
                        match self.stage {
                            St::Vertex => self.output,
                            St::Fragment => !self.output,
                            St::Compute => false,
                        },
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
                    Bi::SampleIndex => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Scalar {
                                kind: Sk::Uint,
                                width,
                            },
                    ),
                    Bi::SampleMask => (
                        self.stage == St::Fragment,
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
            crate::Binding::Location {
                location,
                interpolation,
                sampling,
            } => {
                if !self.location_mask.insert(location as usize) {
                    return Err(VaryingError::BindingCollision { location });
                }

                // Values passed from the vertex shader to the fragment shader must have their
                // interpolation defaulted (i.e. not `None`) by the front end, as appropriate for
                // that language. For anything other than floating-point scalars and vectors, the
                // interpolation must be `Flat`.
                let needs_interpolation = match self.stage {
                    crate::ShaderStage::Vertex => self.output,
                    crate::ShaderStage::Fragment => !self.output,
                    _ => false,
                };

                // It doesn't make sense to specify a sampling when `interpolation` is `Flat`, but
                // SPIR-V and GLSL both explicitly tolerate such combinations of decorators /
                // qualifiers, so we won't complain about that here.
                let _ = sampling;

                match ty_inner.scalar_kind() {
                    Some(crate::ScalarKind::Float) => {
                        if needs_interpolation && interpolation.is_none() {
                            return Err(VaryingError::MissingInterpolation);
                        }
                    }
                    Some(_) => {
                        if needs_interpolation && interpolation != Some(crate::Interpolation::Flat)
                        {
                            return Err(VaryingError::InvalidInterpolation);
                        }
                    }
                    None => return Err(VaryingError::InvalidType(self.ty)),
                }
            }
        }

        Ok(())
    }

    fn validate(&mut self, binding: Option<&crate::Binding>) -> Result<(), VaryingError> {
        match binding {
            Some(binding) => self.validate_impl(binding),
            None => {
                match self.types[self.ty].inner {
                    //TODO: check the member types
                    crate::TypeInner::Struct {
                        top_level: false,
                        ref members,
                        ..
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
                    _ => return Err(VaryingError::MissingBinding),
                }
                Ok(())
            }
        }
    }
}

impl super::Validator {
    pub(super) fn validate_global_var(
        &self,
        var: &crate::GlobalVariable,
        types: &Arena<crate::Type>,
    ) -> Result<(), GlobalVariableError> {
        log::debug!("var {:?}", var);
        let type_info = &self.types[var.ty.index()];

        let (allowed_storage_access, required_type_flags, is_resource) = match var.class {
            crate::StorageClass::Function => return Err(GlobalVariableError::InvalidUsage),
            crate::StorageClass::Storage => {
                if let Err((ty_handle, disalignment)) = type_info.storage_layout {
                    if self.flags.contains(ValidationFlags::STRUCT_LAYOUTS) {
                        return Err(GlobalVariableError::Alignment(ty_handle, disalignment));
                    }
                }
                (
                    crate::StorageAccess::all(),
                    TypeFlags::DATA | TypeFlags::HOST_SHARED | TypeFlags::TOP_LEVEL,
                    true,
                )
            }
            crate::StorageClass::Uniform => {
                if let Err((ty_handle, disalignment)) = type_info.uniform_layout {
                    if self.flags.contains(ValidationFlags::STRUCT_LAYOUTS) {
                        return Err(GlobalVariableError::Alignment(ty_handle, disalignment));
                    }
                }
                (
                    crate::StorageAccess::empty(),
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::HOST_SHARED
                        | TypeFlags::TOP_LEVEL,
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
            crate::StorageClass::Private | crate::StorageClass::WorkGroup => (
                crate::StorageAccess::empty(),
                TypeFlags::DATA | TypeFlags::SIZED,
                false,
            ),
            crate::StorageClass::PushConstant => {
                if !self.capabilities.contains(Capabilities::PUSH_CONSTANT) {
                    return Err(GlobalVariableError::UnsupportedCapability(
                        Capabilities::PUSH_CONSTANT,
                    ));
                }
                (
                    crate::StorageAccess::LOAD,
                    TypeFlags::DATA | TypeFlags::HOST_SHARED | TypeFlags::SIZED,
                    false,
                )
            }
        };

        if !allowed_storage_access.contains(var.storage_access) {
            return Err(GlobalVariableError::InvalidStorageAccess {
                seen: var.storage_access,
                allowed: allowed_storage_access,
            });
        }

        if !type_info.flags.contains(required_type_flags) {
            return Err(GlobalVariableError::MissingTypeFlags {
                seen: type_info.flags,
                required: required_type_flags,
            });
        }

        if is_resource != var.binding.is_some() {
            return Err(GlobalVariableError::InvalidBinding);
        }

        Ok(())
    }

    pub(super) fn validate_entry_point(
        &mut self,
        ep: &crate::EntryPoint,
        module: &crate::Module,
        mod_info: &ModuleInfo,
    ) -> Result<FunctionInfo, EntryPointError> {
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

        let stage_bit = match ep.stage {
            crate::ShaderStage::Vertex => ShaderStages::VERTEX,
            crate::ShaderStage::Fragment => ShaderStages::FRAGMENT,
            crate::ShaderStage::Compute => ShaderStages::COMPUTE,
        };

        let info = self.validate_function(&ep.function, module, mod_info)?;

        if !info.available_stages.contains(stage_bit) {
            return Err(EntryPointError::ForbiddenStageOperations);
        }

        self.location_mask.clear();
        let mut argument_built_ins = 0;
        for (index, fa) in ep.function.arguments.iter().enumerate() {
            let mut ctx = VaryingContext {
                ty: fa.ty,
                stage: ep.stage,
                output: false,
                types: &module.types,
                location_mask: &mut self.location_mask,
                built_in_mask: argument_built_ins,
            };
            ctx.validate(fa.binding.as_ref())
                .map_err(|e| EntryPointError::Argument(index as u32, e))?;
            argument_built_ins = ctx.built_in_mask;
        }

        self.location_mask.clear();
        if let Some(ref fr) = ep.function.result {
            let mut ctx = VaryingContext {
                ty: fr.ty,
                stage: ep.stage,
                output: true,
                types: &module.types,
                location_mask: &mut self.location_mask,
                built_in_mask: 0,
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

        Ok(info)
    }
}
