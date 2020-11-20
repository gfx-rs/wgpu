use super::typifier::{ResolveContext, ResolveError, Typifier};
use crate::arena::{Arena, Handle};

const MAX_BIND_GROUPS: u32 = 8;
const MAX_LOCATIONS: u32 = 64; // using u64 mask
const MAX_BIND_INDICES: u32 = 64; // using u64 mask
const MAX_WORKGROUP_SIZE: u32 = 0x4000;

#[derive(Debug)]
pub struct Validator {
    //Note: this is a bit tricky: some of the front-ends as well as backends
    // already have to use the typifier, so the work here is redundant in a way.
    typifier: Typifier,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum GlobalVariableError {
    #[error("Usage isn't compatible with the storage class")]
    InvalidUsage,
    #[error("Type isn't compatible with the storage class")]
    InvalidType,
    #[error("Interpolation is not valid")]
    InvalidInterpolation,
    #[error("Storage access {seen:?} exceed the allowed {allowed:?}")]
    InvalidStorageAccess {
        allowed: crate::StorageAccess,
        seen: crate::StorageAccess,
    },
    #[error("Binding decoration is missing or not applicable")]
    InvalidBinding,
    #[error("Binding is out of range")]
    OutOfRangeBinding,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LocalVariableError {
    #[error("Initializer is not a constant expression")]
    InitializerConst,
    #[error("Initializer doesn't match the variable type")]
    InitializerType,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FunctionError {
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    #[error("There are instructions after `return`/`break`/`continue`")]
    InvalidControlFlowExitTail,
    #[error("Local variable {handle:?} '{name}' is invalid: {error:?}")]
    LocalVariable {
        handle: Handle<crate::LocalVariable>,
        name: String,
        error: LocalVariableError,
    },
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EntryPointError {
    #[error("Early depth test is not applicable")]
    UnexpectedEarlyDepthTest,
    #[error("Workgroup size is not applicable")]
    UnexpectedWorkgroupSize,
    #[error("Workgroup size is out of range")]
    OutOfRangeWorkgroupSize,
    #[error("Global variable {0:?} is used incorrectly as {1:?}")]
    InvalidGlobalUsage(Handle<crate::GlobalVariable>, crate::GlobalUse),
    #[error("Bindings for {0:?} conflict with other global variables")]
    BindingCollision(Handle<crate::GlobalVariable>),
    #[error("Built-in {0:?} is not applicable to this entry point")]
    InvalidBuiltIn(crate::BuiltIn),
    #[error("Interpolation of an integer has to be flat")]
    InvalidIntegerInterpolation,
    #[error(transparent)]
    Function(#[from] FunctionError),
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ValidationError {
    #[error("The type {0:?} width {1} is not supported")]
    InvalidTypeWidth(crate::ScalarKind, crate::Bytes),
    #[error("The type handle {0:?} can not be resolved")]
    UnresolvedType(Handle<crate::Type>),
    #[error("The constant {0:?} can not be used for an array size")]
    InvalidArraySizeConstant(Handle<crate::Constant>),
    #[error("Global variable {handle:?} '{name}' is invalid: {error:?}")]
    GlobalVariable {
        handle: Handle<crate::GlobalVariable>,
        name: String,
        error: GlobalVariableError,
    },
    #[error("Function {0:?} is invalid: {1:?}")]
    Function(Handle<crate::Function>, FunctionError),
    #[error("Entry point {name} at {stage:?} is invalid: {error:?}")]
    EntryPoint {
        stage: crate::ShaderStage,
        name: String,
        error: EntryPointError,
    },
    #[error("Module is corrupted")]
    Corrupted,
}

impl crate::GlobalVariable {
    fn forbid_interpolation(&self) -> Result<(), GlobalVariableError> {
        match self.interpolation {
            Some(_) => Err(GlobalVariableError::InvalidInterpolation),
            None => Ok(()),
        }
    }

    fn check_resource(&self) -> Result<(), GlobalVariableError> {
        match self.binding {
            Some(crate::Binding::BuiltIn(_)) => {} // validated per entry point
            Some(crate::Binding::Resource { group, binding }) => {
                if group > MAX_BIND_GROUPS || binding > MAX_BIND_INDICES {
                    return Err(GlobalVariableError::OutOfRangeBinding);
                }
            }
            Some(crate::Binding::Location(_)) | None => {
                return Err(GlobalVariableError::InvalidBinding)
            }
        }
        self.forbid_interpolation()
    }
}

fn storage_usage(access: crate::StorageAccess) -> crate::GlobalUse {
    let mut storage_usage = crate::GlobalUse::empty();
    if access.contains(crate::StorageAccess::LOAD) {
        storage_usage |= crate::GlobalUse::LOAD;
    }
    if access.contains(crate::StorageAccess::STORE) {
        storage_usage |= crate::GlobalUse::STORE;
    }
    storage_usage
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new() -> Self {
        Validator {
            typifier: Typifier::new(),
        }
    }

    fn validate_global_var(
        &self,
        var: &crate::GlobalVariable,
        types: &Arena<crate::Type>,
    ) -> Result<(), GlobalVariableError> {
        log::debug!("var {:?}", var);
        let allowed_storage_access = match var.class {
            crate::StorageClass::Function => return Err(GlobalVariableError::InvalidUsage),
            crate::StorageClass::Input | crate::StorageClass::Output => {
                match var.binding {
                    Some(crate::Binding::BuiltIn(_)) => {
                        // validated per entry point
                        var.forbid_interpolation()?
                    }
                    Some(crate::Binding::Location(loc)) => {
                        if loc > MAX_LOCATIONS {
                            return Err(GlobalVariableError::OutOfRangeBinding);
                        }
                        match types[var.ty].inner {
                            crate::TypeInner::Scalar { .. }
                            | crate::TypeInner::Vector { .. }
                            | crate::TypeInner::Matrix { .. } => {}
                            _ => return Err(GlobalVariableError::InvalidType),
                        }
                    }
                    Some(crate::Binding::Resource { .. }) => {
                        return Err(GlobalVariableError::InvalidBinding)
                    }
                    None => {
                        match types[var.ty].inner {
                            //TODO: check the member types
                            crate::TypeInner::Struct { members: _ } => {
                                var.forbid_interpolation()?
                            }
                            _ => return Err(GlobalVariableError::InvalidType),
                        }
                    }
                }
                crate::StorageAccess::empty()
            }
            crate::StorageClass::Storage => {
                var.check_resource()?;
                crate::StorageAccess::all()
            }
            crate::StorageClass::Uniform => {
                var.check_resource()?;
                crate::StorageAccess::empty()
            }
            crate::StorageClass::Handle => {
                var.check_resource()?;
                match types[var.ty].inner {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage(_),
                        ..
                    } => crate::StorageAccess::all(),
                    _ => crate::StorageAccess::empty(),
                }
            }
            crate::StorageClass::Private | crate::StorageClass::WorkGroup => {
                if var.binding.is_some() {
                    return Err(GlobalVariableError::InvalidBinding);
                }
                var.forbid_interpolation()?;
                crate::StorageAccess::empty()
            }
            crate::StorageClass::PushConstant => {
                //TODO
                return Err(GlobalVariableError::InvalidStorageAccess {
                    allowed: crate::StorageAccess::empty(),
                    seen: crate::StorageAccess::empty(),
                });
            }
        };

        if !allowed_storage_access.contains(var.storage_access) {
            return Err(GlobalVariableError::InvalidStorageAccess {
                allowed: allowed_storage_access,
                seen: var.storage_access,
            });
        }

        Ok(())
    }

    fn validate_local_var(
        &self,
        var: &crate::LocalVariable,
        _fun: &crate::Function,
        _types: &Arena<crate::Type>,
    ) -> Result<(), LocalVariableError> {
        log::debug!("var {:?}", var);
        if let Some(_expr_handle) = var.init {
            if false {
                return Err(LocalVariableError::InitializerConst);
            }
        }
        Ok(())
    }

    fn validate_function(
        &mut self,
        fun: &crate::Function,
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
            self.validate_local_var(var, fun, &module.types)
                .map_err(|error| FunctionError::LocalVariable {
                    handle: var_handle,
                    name: var.name.clone().unwrap_or_default(),
                    error,
                })?;
        }
        Ok(())
    }

    fn validate_entry_point(
        &mut self,
        ep: &crate::EntryPoint,
        stage: crate::ShaderStage,
        module: &crate::Module,
    ) -> Result<(), EntryPointError> {
        if ep.early_depth_test.is_some() && stage != crate::ShaderStage::Fragment {
            return Err(EntryPointError::UnexpectedEarlyDepthTest);
        }
        if stage == crate::ShaderStage::Compute {
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

        let mut bind_group_masks = [0u64; MAX_BIND_GROUPS as usize];
        let mut location_in_mask = 0u64;
        let mut location_out_mask = 0u64;
        for ((var_handle, var), &usage) in module
            .global_variables
            .iter()
            .zip(&ep.function.global_usage)
        {
            if usage.is_empty() {
                continue;
            }

            if let Some(crate::Binding::Location(_)) = var.binding {
                match (stage, var.class) {
                    (crate::ShaderStage::Vertex, crate::StorageClass::Output)
                    | (crate::ShaderStage::Fragment, crate::StorageClass::Input) => {
                        match module.types[var.ty].inner.scalar_kind() {
                            Some(crate::ScalarKind::Float) => {}
                            Some(_) if var.interpolation != Some(crate::Interpolation::Flat) => {
                                return Err(EntryPointError::InvalidIntegerInterpolation);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }

            let allowed_usage = match var.class {
                crate::StorageClass::Function => unreachable!(),
                crate::StorageClass::Input => {
                    let mask = match var.binding {
                        Some(crate::Binding::BuiltIn(built_in)) => match (stage, built_in) {
                            (crate::ShaderStage::Vertex, crate::BuiltIn::BaseInstance)
                            | (crate::ShaderStage::Vertex, crate::BuiltIn::BaseVertex)
                            | (crate::ShaderStage::Vertex, crate::BuiltIn::InstanceIndex)
                            | (crate::ShaderStage::Vertex, crate::BuiltIn::VertexIndex)
                            | (crate::ShaderStage::Fragment, crate::BuiltIn::PointSize)
                            | (crate::ShaderStage::Fragment, crate::BuiltIn::FragCoord)
                            | (crate::ShaderStage::Fragment, crate::BuiltIn::FrontFacing)
                            | (crate::ShaderStage::Fragment, crate::BuiltIn::SampleIndex)
                            | (crate::ShaderStage::Compute, crate::BuiltIn::GlobalInvocationId)
                            | (crate::ShaderStage::Compute, crate::BuiltIn::LocalInvocationId)
                            | (crate::ShaderStage::Compute, crate::BuiltIn::LocalInvocationIndex)
                            | (crate::ShaderStage::Compute, crate::BuiltIn::WorkGroupId) => 0,
                            _ => return Err(EntryPointError::InvalidBuiltIn(built_in)),
                        },
                        Some(crate::Binding::Location(loc)) => 1 << loc,
                        Some(crate::Binding::Resource { .. }) => unreachable!(),
                        None => 0,
                    };
                    if location_in_mask & mask != 0 {
                        return Err(EntryPointError::BindingCollision(var_handle));
                    }
                    location_in_mask |= mask;
                    crate::GlobalUse::LOAD
                }
                crate::StorageClass::Output => {
                    let mask = match var.binding {
                        Some(crate::Binding::BuiltIn(built_in)) => match (stage, built_in) {
                            (crate::ShaderStage::Vertex, crate::BuiltIn::Position)
                            | (crate::ShaderStage::Vertex, crate::BuiltIn::PointSize)
                            | (crate::ShaderStage::Vertex, crate::BuiltIn::ClipDistance)
                            | (crate::ShaderStage::Fragment, crate::BuiltIn::FragDepth) => 0,
                            _ => return Err(EntryPointError::InvalidBuiltIn(built_in)),
                        },
                        Some(crate::Binding::Location(loc)) => 1 << loc,
                        Some(crate::Binding::Resource { .. }) => unreachable!(),
                        None => 0,
                    };
                    if location_out_mask & mask != 0 {
                        return Err(EntryPointError::BindingCollision(var_handle));
                    }
                    location_out_mask |= mask;
                    crate::GlobalUse::LOAD | crate::GlobalUse::STORE
                }
                crate::StorageClass::Uniform => crate::GlobalUse::LOAD,
                crate::StorageClass::Storage => storage_usage(var.storage_access),
                crate::StorageClass::Handle => match module.types[var.ty].inner {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage(_),
                        ..
                    } => storage_usage(var.storage_access),
                    _ => crate::GlobalUse::LOAD,
                },
                crate::StorageClass::Private | crate::StorageClass::WorkGroup => {
                    crate::GlobalUse::all()
                }
                crate::StorageClass::PushConstant => crate::GlobalUse::LOAD,
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

            if let Some(crate::Binding::Resource { group, binding }) = var.binding {
                let mask = 1 << binding;
                let group_mask = &mut bind_group_masks[group as usize];
                if *group_mask & mask != 0 {
                    return Err(EntryPointError::BindingCollision(var_handle));
                }
                *group_mask |= mask;
            }
        }

        self.validate_function(&ep.function, module)?;
        Ok(())
    }

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<(), ValidationError> {
        // check the types
        for (handle, ty) in module.types.iter() {
            use crate::TypeInner as Ti;
            match ty.inner {
                Ti::Scalar { kind, width } | Ti::Vector { kind, width, .. } => {
                    let expected = match kind {
                        crate::ScalarKind::Bool => 1,
                        _ => 4,
                    };
                    if width != expected {
                        return Err(ValidationError::InvalidTypeWidth(kind, width));
                    }
                }
                Ti::Matrix { width, .. } => {
                    if width != 4 {
                        return Err(ValidationError::InvalidTypeWidth(
                            crate::ScalarKind::Float,
                            width,
                        ));
                    }
                }
                Ti::Pointer { base, class: _ } => {
                    if base >= handle {
                        return Err(ValidationError::UnresolvedType(base));
                    }
                }
                Ti::Array { base, size, .. } => {
                    if base >= handle {
                        return Err(ValidationError::UnresolvedType(base));
                    }
                    if let crate::ArraySize::Constant(const_handle) = size {
                        let constant = module
                            .constants
                            .try_get(const_handle)
                            .ok_or(ValidationError::Corrupted)?;
                        match constant.inner {
                            crate::ConstantInner::Uint(_) => {}
                            _ => {
                                return Err(ValidationError::InvalidArraySizeConstant(const_handle))
                            }
                        }
                    }
                }
                Ti::Struct { ref members } => {
                    //TODO: check that offsets are not intersecting?
                    for member in members {
                        if member.ty >= handle {
                            return Err(ValidationError::UnresolvedType(member.ty));
                        }
                    }
                }
                Ti::Image { .. } => {}
                Ti::Sampler { comparison: _ } => {}
            }
        }

        for (var_handle, var) in module.global_variables.iter() {
            self.validate_global_var(var, &module.types)
                .map_err(|error| ValidationError::GlobalVariable {
                    handle: var_handle,
                    name: var.name.clone().unwrap_or_default(),
                    error,
                })?;
        }

        for (fun_handle, fun) in module.functions.iter() {
            self.validate_function(fun, module)
                .map_err(|e| ValidationError::Function(fun_handle, e))?;
        }

        for (&(stage, ref name), entry_point) in module.entry_points.iter() {
            self.validate_entry_point(entry_point, stage, module)
                .map_err(|error| ValidationError::EntryPoint {
                    stage,
                    name: name.to_string(),
                    error,
                })?;
        }

        Ok(())
    }
}
