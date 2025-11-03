use alloc::vec::Vec;

use bit_set::BitSet;

use super::{
    analyzer::{FunctionInfo, GlobalUse},
    Capabilities, Disalignment, FunctionError, ModuleInfo, PushConstantError,
};
use crate::arena::{Handle, UniqueArena};
use crate::span::{AddSpan as _, MapErrWithSpan as _, SpanProvider as _, WithSpan};

const MAX_WORKGROUP_SIZE: u32 = 0x4000;

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum GlobalVariableError {
    #[error("Usage isn't compatible with address space {0:?}")]
    InvalidUsage(crate::AddressSpace),
    #[error("Type isn't compatible with address space {0:?}")]
    InvalidType(crate::AddressSpace),
    #[error("Type flags {seen:?} do not meet the required {required:?}")]
    MissingTypeFlags {
        required: super::TypeFlags,
        seen: super::TypeFlags,
    },
    #[error("Capability {0:?} is not supported")]
    UnsupportedCapability(Capabilities),
    #[error("Binding decoration is missing or not applicable")]
    InvalidBinding,
    #[error("Alignment requirements for address space {0:?} are not met by {1:?}")]
    Alignment(
        crate::AddressSpace,
        Handle<crate::Type>,
        #[source] Disalignment,
    ),
    #[error("Initializer must be an override-expression")]
    InitializerExprType,
    #[error("Initializer doesn't match the variable type")]
    InitializerType,
    #[error("Initializer can't be used with address space {0:?}")]
    InitializerNotAllowed(crate::AddressSpace),
    #[error("Storage address space doesn't support write-only access")]
    StorageAddressSpaceWriteOnlyNotSupported,
    #[error("Type is not valid for use as a push constant")]
    InvalidPushConstantType(#[source] PushConstantError),
    #[error("Task payload must not be zero-sized")]
    ZeroSizedTaskPayload,
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum VaryingError {
    #[error("The type {0:?} does not match the varying")]
    InvalidType(Handle<crate::Type>),
    #[error("The type {0:?} cannot be used for user-defined entry point inputs or outputs")]
    NotIOShareableType(Handle<crate::Type>),
    #[error("Interpolation is not valid")]
    InvalidInterpolation,
    #[error("Cannot combine {interpolation:?} interpolation with the {sampling:?} sample type")]
    InvalidInterpolationSamplingCombination {
        interpolation: crate::Interpolation,
        sampling: crate::Sampling,
    },
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
    #[error("Multiple bindings use the same `blend_src` {blend_src}")]
    BindingCollisionBlendSrc { blend_src: u32 },
    #[error("Built-in {0:?} is present more than once")]
    DuplicateBuiltIn(crate::BuiltIn),
    #[error("Capability {0:?} is not supported")]
    UnsupportedCapability(Capabilities),
    #[error("The attribute {0:?} is only valid as an output for stage {1:?}")]
    InvalidInputAttributeInStage(&'static str, crate::ShaderStage),
    #[error("The attribute {0:?} is not valid for stage {1:?}")]
    InvalidAttributeInStage(&'static str, crate::ShaderStage),
    #[error("The `blend_src` attribute can only be used on location 0, only indices 0 and 1 are valid. Location was {location}, index was {blend_src}.")]
    InvalidBlendSrcIndex { location: u32, blend_src: u32 },
    #[error("If `blend_src` is used, there must be exactly two outputs both with location 0, one with `blend_src(0)` and the other with `blend_src(1)`.")]
    IncompleteBlendSrcUsage,
    #[error("If `blend_src` is used, both outputs must have the same type. `blend_src(0)` has type {blend_src_0_type:?} and `blend_src(1)` has type {blend_src_1_type:?}.")]
    BlendSrcOutputTypeMismatch {
        blend_src_0_type: Handle<crate::Type>,
        blend_src_1_type: Handle<crate::Type>,
    },
    #[error("Workgroup size is multi dimensional, `@builtin(subgroup_id)` and `@builtin(subgroup_invocation_id)` are not supported.")]
    InvalidMultiDimensionalSubgroupBuiltIn,
    #[error("The `@per_primitive` attribute can only be used in fragment shader inputs or mesh shader primitive outputs")]
    InvalidPerPrimitive,
    #[error("Non-builtin members of a mesh primitive output struct must be decorated with `@per_primitive`")]
    MissingPerPrimitive,
    #[error("The `MESH_SHADER` capability must be enabled to use per-primitive fragment inputs.")]
    PerPrimitiveNotAllowed,
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum EntryPointError {
    #[error("Multiple conflicting entry points")]
    Conflict,
    #[error("Vertex shaders must return a `@builtin(position)` output value")]
    MissingVertexOutputPosition,
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
    #[error("More than 1 push constant variable is used")]
    MoreThanOnePushConstantUsed,
    #[error("Bindings for {0:?} conflict with other resource")]
    BindingCollision(Handle<crate::GlobalVariable>),
    #[error("Argument {0} varying error")]
    Argument(u32, #[source] VaryingError),
    #[error(transparent)]
    Result(#[from] VaryingError),
    #[error("Location {location} interpolation of an integer has to be flat")]
    InvalidIntegerInterpolation { location: u32 },
    #[error(transparent)]
    Function(#[from] FunctionError),
    #[error("mesh shader entry point missing mesh shader attributes")]
    ExpectedMeshShaderAttributes,
    #[error("Non mesh shader entry point cannot have mesh shader attributes")]
    UnexpectedMeshShaderAttributes,
    #[error("Non mesh/task shader entry point cannot have task payload attribute")]
    UnexpectedTaskPayload,
    #[error("Task payload must be declared with `var<task_payload>`")]
    TaskPayloadWrongAddressSpace,
    #[error("For a task payload to be used, it must be declared with @payload")]
    WrongTaskPayloadUsed,
    #[error("A function can only set vertex and primitive types that correspond to the mesh shader attributes")]
    WrongMeshOutputType,
    #[error("Only mesh shader entry points can write to mesh output vertices and primitives")]
    UnexpectedMeshShaderOutput,
    #[error("Mesh shader entry point cannot have a return type")]
    UnexpectedMeshShaderEntryResult,
    #[error("Task shader entry point must return @builtin(mesh_task_size) vec3<u32>")]
    WrongTaskShaderEntryResult,
    #[error("Mesh output type must be a user-defined struct.")]
    InvalidMeshOutputType,
    #[error("Mesh primitive outputs must have exactly one of `@builtin(triangle_indices)`, `@builtin(line_indices)`, or `@builtin(point_index)`")]
    InvalidMeshPrimitiveOutputType,
    #[error("Task shaders must declare a task payload output")]
    ExpectedTaskPayload,
    #[error(
        "The `MESH_SHADER` capability must be enabled to compile mesh shaders and task shaders."
    )]
    MeshShaderCapabilityDisabled,
}

fn storage_usage(access: crate::StorageAccess) -> GlobalUse {
    let mut storage_usage = GlobalUse::QUERY;
    if access.contains(crate::StorageAccess::LOAD) {
        storage_usage |= GlobalUse::READ;
    }
    if access.contains(crate::StorageAccess::STORE) {
        storage_usage |= GlobalUse::WRITE;
    }
    if access.contains(crate::StorageAccess::ATOMIC) {
        storage_usage |= GlobalUse::ATOMIC;
    }
    storage_usage
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshOutputType {
    None,
    VertexOutput,
    PrimitiveOutput,
}

struct VaryingContext<'a> {
    stage: crate::ShaderStage,
    output: bool,
    types: &'a UniqueArena<crate::Type>,
    type_info: &'a Vec<super::r#type::TypeInfo>,
    location_mask: &'a mut BitSet,
    blend_src_mask: &'a mut BitSet,
    built_ins: &'a mut crate::FastHashSet<crate::BuiltIn>,
    capabilities: Capabilities,
    flags: super::ValidationFlags,
    mesh_output_type: MeshOutputType,
    has_task_payload: bool,
}

impl VaryingContext<'_> {
    fn validate_impl(
        &mut self,
        ep: &crate::EntryPoint,
        ty: Handle<crate::Type>,
        binding: &crate::Binding,
    ) -> Result<(), VaryingError> {
        use crate::{BuiltIn as Bi, ShaderStage as St, TypeInner as Ti, VectorSize as Vs};

        let ty_inner = &self.types[ty].inner;
        match *binding {
            crate::Binding::BuiltIn(built_in) => {
                // Ignore the `invariant` field for the sake of duplicate checks,
                // but use the original in error messages.
                let canonical = if let crate::BuiltIn::Position { .. } = built_in {
                    crate::BuiltIn::Position { invariant: false }
                } else {
                    built_in
                };

                if self.built_ins.contains(&canonical) {
                    return Err(VaryingError::DuplicateBuiltIn(built_in));
                }
                self.built_ins.insert(canonical);

                let required = match built_in {
                    Bi::ClipDistance => Capabilities::CLIP_DISTANCE,
                    Bi::CullDistance => Capabilities::CULL_DISTANCE,
                    Bi::PrimitiveIndex => Capabilities::PRIMITIVE_INDEX,
                    Bi::Barycentric => Capabilities::SHADER_BARYCENTRICS,
                    Bi::ViewIndex => Capabilities::MULTIVIEW,
                    Bi::SampleIndex => Capabilities::MULTISAMPLED_SHADING,
                    Bi::NumSubgroups
                    | Bi::SubgroupId
                    | Bi::SubgroupSize
                    | Bi::SubgroupInvocationId => Capabilities::SUBGROUP,
                    _ => Capabilities::empty(),
                };
                if !self.capabilities.contains(required) {
                    return Err(VaryingError::UnsupportedCapability(required));
                }

                if matches!(
                    built_in,
                    crate::BuiltIn::SubgroupId | crate::BuiltIn::SubgroupInvocationId
                ) && ep.workgroup_size[1..].iter().any(|&s| s > 1)
                {
                    return Err(VaryingError::InvalidMultiDimensionalSubgroupBuiltIn);
                }

                let (visible, type_good) = match built_in {
                    Bi::BaseInstance | Bi::BaseVertex | Bi::InstanceIndex | Bi::VertexIndex => (
                        self.stage == St::Vertex && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::DrawID => (
                        // Always allowed in task/vertex stage. Allowed in mesh stage if there is no task stage in the pipeline.
                        (self.stage == St::Vertex
                            || self.stage == St::Task
                            || (self.stage == St::Mesh && !self.has_task_payload))
                            && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::ClipDistance | Bi::CullDistance => (
                        (self.stage == St::Vertex || self.stage == St::Mesh) && self.output,
                        match *ty_inner {
                            Ti::Array { base, size, .. } => {
                                self.types[base].inner == Ti::Scalar(crate::Scalar::F32)
                                    && match size {
                                        crate::ArraySize::Constant(non_zero) => non_zero.get() <= 8,
                                        _ => false,
                                    }
                            }
                            _ => false,
                        },
                    ),
                    Bi::PointSize => (
                        (self.stage == St::Vertex || self.stage == St::Mesh) && self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::F32),
                    ),
                    Bi::PointCoord => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Bi,
                                scalar: crate::Scalar::F32,
                            },
                    ),
                    Bi::Position { .. } => (
                        match self.stage {
                            St::Vertex | St::Mesh => self.output,
                            St::Fragment => !self.output,
                            St::Compute | St::Task => false,
                        },
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Quad,
                                scalar: crate::Scalar::F32,
                            },
                    ),
                    Bi::ViewIndex => (
                        match self.stage {
                            St::Vertex | St::Fragment | St::Task | St::Mesh => !self.output,
                            St::Compute => false,
                        },
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::FragDepth => (
                        self.stage == St::Fragment && self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::F32),
                    ),
                    Bi::FrontFacing => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::BOOL),
                    ),
                    Bi::PrimitiveIndex => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::Barycentric => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Tri,
                                scalar: crate::Scalar::F32,
                            },
                    ),
                    Bi::SampleIndex => (
                        self.stage == St::Fragment && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::SampleMask => (
                        self.stage == St::Fragment,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::LocalInvocationIndex => (
                        self.stage.compute_like() && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::GlobalInvocationId
                    | Bi::LocalInvocationId
                    | Bi::WorkGroupId
                    | Bi::WorkGroupSize
                    | Bi::NumWorkGroups => (
                        self.stage.compute_like() && !self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Tri,
                                scalar: crate::Scalar::U32,
                            },
                    ),
                    Bi::NumSubgroups | Bi::SubgroupId => (
                        self.stage.compute_like() && !self.output,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::SubgroupSize | Bi::SubgroupInvocationId => (
                        match self.stage {
                            St::Compute | St::Fragment | St::Task | St::Mesh => !self.output,
                            St::Vertex => false,
                        },
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::CullPrimitive => (
                        self.mesh_output_type == MeshOutputType::PrimitiveOutput,
                        *ty_inner == Ti::Scalar(crate::Scalar::BOOL),
                    ),
                    Bi::PointIndex => (
                        self.mesh_output_type == MeshOutputType::PrimitiveOutput,
                        *ty_inner == Ti::Scalar(crate::Scalar::U32),
                    ),
                    Bi::LineIndices => (
                        self.mesh_output_type == MeshOutputType::PrimitiveOutput,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Bi,
                                scalar: crate::Scalar::U32,
                            },
                    ),
                    Bi::TriangleIndices => (
                        self.mesh_output_type == MeshOutputType::PrimitiveOutput,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Tri,
                                scalar: crate::Scalar::U32,
                            },
                    ),
                    Bi::MeshTaskSize => (
                        self.stage == St::Task && self.output,
                        *ty_inner
                            == Ti::Vector {
                                size: Vs::Tri,
                                scalar: crate::Scalar::U32,
                            },
                    ),
                };

                if !visible {
                    return Err(VaryingError::InvalidBuiltInStage(built_in));
                }
                if !type_good {
                    log::warn!("Wrong builtin type: {ty_inner:?}");
                    return Err(VaryingError::InvalidBuiltInType(built_in));
                }
            }
            crate::Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                if per_primitive && !self.capabilities.contains(Capabilities::MESH_SHADER) {
                    return Err(VaryingError::PerPrimitiveNotAllowed);
                }
                // Only IO-shareable types may be stored in locations.
                if !self.type_info[ty.index()]
                    .flags
                    .contains(super::TypeFlags::IO_SHAREABLE)
                {
                    return Err(VaryingError::NotIOShareableType(ty));
                }

                // Check whether `per_primitive` is appropriate for this stage and direction.
                if self.mesh_output_type == MeshOutputType::PrimitiveOutput {
                    // All mesh shader `Location` outputs must be `per_primitive`.
                    if !per_primitive {
                        return Err(VaryingError::MissingPerPrimitive);
                    }
                } else if self.stage == crate::ShaderStage::Fragment && !self.output {
                    // Fragment stage inputs may be `per_primitive`. We'll only
                    // know if these are correct when the whole mesh pipeline is
                    // created and we're paired with a specific mesh or vertex
                    // shader.
                } else if per_primitive {
                    // All other `Location` bindings must not be `per_primitive`.
                    return Err(VaryingError::InvalidPerPrimitive);
                }

                if let Some(blend_src) = blend_src {
                    // `blend_src` is only valid if dual source blending was explicitly enabled,
                    // see https://www.w3.org/TR/WGSL/#extension-dual_source_blending
                    if !self
                        .capabilities
                        .contains(Capabilities::DUAL_SOURCE_BLENDING)
                    {
                        return Err(VaryingError::UnsupportedCapability(
                            Capabilities::DUAL_SOURCE_BLENDING,
                        ));
                    }
                    if self.stage != crate::ShaderStage::Fragment {
                        return Err(VaryingError::InvalidAttributeInStage(
                            "blend_src",
                            self.stage,
                        ));
                    }
                    if !self.output {
                        return Err(VaryingError::InvalidInputAttributeInStage(
                            "blend_src",
                            self.stage,
                        ));
                    }
                    if (blend_src != 0 && blend_src != 1) || location != 0 {
                        return Err(VaryingError::InvalidBlendSrcIndex {
                            location,
                            blend_src,
                        });
                    }
                    if !self.blend_src_mask.insert(blend_src as usize) {
                        return Err(VaryingError::BindingCollisionBlendSrc { blend_src });
                    }
                } else if !self.location_mask.insert(location as usize)
                    && self.flags.contains(super::ValidationFlags::BINDINGS)
                {
                    return Err(VaryingError::BindingCollision { location });
                }

                if let Some(interpolation) = interpolation {
                    let invalid_sampling = match (interpolation, sampling) {
                        (_, None)
                        | (
                            crate::Interpolation::Perspective | crate::Interpolation::Linear,
                            Some(
                                crate::Sampling::Center
                                | crate::Sampling::Centroid
                                | crate::Sampling::Sample,
                            ),
                        )
                        | (
                            crate::Interpolation::Flat,
                            Some(crate::Sampling::First | crate::Sampling::Either),
                        ) => None,
                        (_, Some(invalid_sampling)) => Some(invalid_sampling),
                    };
                    if let Some(sampling) = invalid_sampling {
                        return Err(VaryingError::InvalidInterpolationSamplingCombination {
                            interpolation,
                            sampling,
                        });
                    }
                }

                let needs_interpolation = match self.stage {
                    crate::ShaderStage::Vertex => self.output,
                    crate::ShaderStage::Fragment => !self.output && !per_primitive,
                    crate::ShaderStage::Compute | crate::ShaderStage::Task => false,
                    crate::ShaderStage::Mesh => self.output,
                };

                // It doesn't make sense to specify a sampling when `interpolation` is `Flat`, but
                // SPIR-V and GLSL both explicitly tolerate such combinations of decorators /
                // qualifiers, so we won't complain about that here.
                let _ = sampling;

                let required = match sampling {
                    Some(crate::Sampling::Sample) => Capabilities::MULTISAMPLED_SHADING,
                    _ => Capabilities::empty(),
                };
                if !self.capabilities.contains(required) {
                    return Err(VaryingError::UnsupportedCapability(required));
                }

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
                    None => return Err(VaryingError::InvalidType(ty)),
                }
            }
        }

        Ok(())
    }

    fn validate(
        &mut self,
        ep: &crate::EntryPoint,
        ty: Handle<crate::Type>,
        binding: Option<&crate::Binding>,
    ) -> Result<(), WithSpan<VaryingError>> {
        let span_context = self.types.get_span_context(ty);
        match binding {
            Some(binding) => self
                .validate_impl(ep, ty, binding)
                .map_err(|e| e.with_span_context(span_context)),
            None => {
                match self.types[ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        for (index, member) in members.iter().enumerate() {
                            let span_context = self.types.get_span_context(ty);
                            match member.binding {
                                None => {
                                    if self.flags.contains(super::ValidationFlags::BINDINGS) {
                                        return Err(VaryingError::MemberMissingBinding(
                                            index as u32,
                                        )
                                        .with_span_context(span_context));
                                    }
                                }
                                Some(ref binding) => self
                                    .validate_impl(ep, member.ty, binding)
                                    .map_err(|e| e.with_span_context(span_context))?,
                            }
                        }

                        if !self.blend_src_mask.is_empty() {
                            let span_context = self.types.get_span_context(ty);

                            // If there's any blend_src usage, it must apply to all members of which there must be exactly 2.
                            if members.len() != 2 || self.blend_src_mask.len() != 2 {
                                return Err(VaryingError::IncompleteBlendSrcUsage
                                    .with_span_context(span_context));
                            }
                            // Also, all members must have the same type.
                            if members[0].ty != members[1].ty {
                                return Err(VaryingError::BlendSrcOutputTypeMismatch {
                                    blend_src_0_type: members[0].ty,
                                    blend_src_1_type: members[1].ty,
                                }
                                .with_span_context(span_context));
                            }
                        }
                    }
                    _ => {
                        if self.flags.contains(super::ValidationFlags::BINDINGS) {
                            return Err(VaryingError::MissingBinding.with_span());
                        }
                    }
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
        gctx: crate::proc::GlobalCtx,
        mod_info: &ModuleInfo,
        global_expr_kind: &crate::proc::ExpressionKindTracker,
    ) -> Result<(), GlobalVariableError> {
        use super::TypeFlags;

        log::debug!("var {var:?}");
        let inner_ty = match gctx.types[var.ty].inner {
            // A binding array is (mostly) supposed to behave the same as a
            // series of individually bound resources, so we can (mostly)
            // validate a `binding_array<T>` as if it were just a plain `T`.
            crate::TypeInner::BindingArray { base, .. } => match var.space {
                crate::AddressSpace::Storage { .. }
                | crate::AddressSpace::Uniform
                | crate::AddressSpace::Handle => base,
                _ => return Err(GlobalVariableError::InvalidUsage(var.space)),
            },
            _ => var.ty,
        };
        let type_info = &self.types[inner_ty.index()];

        let (required_type_flags, is_resource) = match var.space {
            crate::AddressSpace::Function => {
                return Err(GlobalVariableError::InvalidUsage(var.space))
            }
            crate::AddressSpace::Storage { access } => {
                if let Err((ty_handle, disalignment)) = type_info.storage_layout {
                    if self.flags.contains(super::ValidationFlags::STRUCT_LAYOUTS) {
                        return Err(GlobalVariableError::Alignment(
                            var.space,
                            ty_handle,
                            disalignment,
                        ));
                    }
                }
                if access == crate::StorageAccess::STORE {
                    return Err(GlobalVariableError::StorageAddressSpaceWriteOnlyNotSupported);
                }
                (
                    TypeFlags::DATA | TypeFlags::HOST_SHAREABLE | TypeFlags::CREATION_RESOLVED,
                    true,
                )
            }
            crate::AddressSpace::Uniform => {
                if let Err((ty_handle, disalignment)) = type_info.uniform_layout {
                    if self.flags.contains(super::ValidationFlags::STRUCT_LAYOUTS) {
                        return Err(GlobalVariableError::Alignment(
                            var.space,
                            ty_handle,
                            disalignment,
                        ));
                    }
                }
                (
                    TypeFlags::DATA
                        | TypeFlags::COPY
                        | TypeFlags::SIZED
                        | TypeFlags::HOST_SHAREABLE
                        | TypeFlags::CREATION_RESOLVED,
                    true,
                )
            }
            crate::AddressSpace::Handle => {
                match gctx.types[inner_ty].inner {
                    crate::TypeInner::Image { class, .. } => match class {
                        crate::ImageClass::Storage {
                            format:
                                crate::StorageFormat::R16Unorm
                                | crate::StorageFormat::R16Snorm
                                | crate::StorageFormat::Rg16Unorm
                                | crate::StorageFormat::Rg16Snorm
                                | crate::StorageFormat::Rgba16Unorm
                                | crate::StorageFormat::Rgba16Snorm,
                            ..
                        } => {
                            if !self
                                .capabilities
                                .contains(Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS)
                            {
                                return Err(GlobalVariableError::UnsupportedCapability(
                                    Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS,
                                ));
                            }
                        }
                        _ => {}
                    },
                    crate::TypeInner::Sampler { .. }
                    | crate::TypeInner::AccelerationStructure { .. }
                    | crate::TypeInner::RayQuery { .. } => {}
                    _ => {
                        return Err(GlobalVariableError::InvalidType(var.space));
                    }
                }

                (TypeFlags::empty(), true)
            }
            crate::AddressSpace::Private => (
                TypeFlags::CONSTRUCTIBLE | TypeFlags::CREATION_RESOLVED,
                false,
            ),
            crate::AddressSpace::WorkGroup => (TypeFlags::DATA | TypeFlags::SIZED, false),
            crate::AddressSpace::TaskPayload => {
                if !self.capabilities.contains(Capabilities::MESH_SHADER) {
                    return Err(GlobalVariableError::UnsupportedCapability(
                        Capabilities::MESH_SHADER,
                    ));
                }
                (TypeFlags::DATA | TypeFlags::SIZED, false)
            }
            crate::AddressSpace::PushConstant => {
                if !self.capabilities.contains(Capabilities::PUSH_CONSTANT) {
                    return Err(GlobalVariableError::UnsupportedCapability(
                        Capabilities::PUSH_CONSTANT,
                    ));
                }
                if let Err(ref err) = type_info.push_constant_compatibility {
                    return Err(GlobalVariableError::InvalidPushConstantType(err.clone()));
                }
                (
                    TypeFlags::DATA
                        | TypeFlags::COPY
                        | TypeFlags::HOST_SHAREABLE
                        | TypeFlags::SIZED,
                    false,
                )
            }
        };

        if !type_info.flags.contains(required_type_flags) {
            return Err(GlobalVariableError::MissingTypeFlags {
                seen: type_info.flags,
                required: required_type_flags,
            });
        }

        if is_resource != var.binding.is_some() {
            if self.flags.contains(super::ValidationFlags::BINDINGS) {
                return Err(GlobalVariableError::InvalidBinding);
            }
        }

        if var.space == crate::AddressSpace::TaskPayload {
            let ty = &gctx.types[var.ty].inner;
            // HLSL doesn't allow zero sized payloads.
            if ty.try_size(gctx) == Some(0) {
                return Err(GlobalVariableError::ZeroSizedTaskPayload);
            }
        }

        if let Some(init) = var.init {
            match var.space {
                crate::AddressSpace::Private | crate::AddressSpace::Function => {}
                _ => {
                    return Err(GlobalVariableError::InitializerNotAllowed(var.space));
                }
            }

            if !global_expr_kind.is_const_or_override(init) {
                return Err(GlobalVariableError::InitializerExprType);
            }

            if !gctx.compare_types(
                &crate::proc::TypeResolution::Handle(var.ty),
                &mod_info[init],
            ) {
                return Err(GlobalVariableError::InitializerType);
            }
        }

        Ok(())
    }

    /// Validate the mesh shader output type `ty`, used as `mesh_output_type`.
    fn validate_mesh_output_type(
        &mut self,
        ep: &crate::EntryPoint,
        module: &crate::Module,
        ty: Handle<crate::Type>,
        mesh_output_type: MeshOutputType,
    ) -> Result<(), WithSpan<EntryPointError>> {
        if !matches!(module.types[ty].inner, crate::TypeInner::Struct { .. }) {
            return Err(EntryPointError::InvalidMeshOutputType.with_span_handle(ty, &module.types));
        }
        let mut result_built_ins = crate::FastHashSet::default();
        let mut ctx = VaryingContext {
            stage: ep.stage,
            output: true,
            types: &module.types,
            type_info: &self.types,
            location_mask: &mut self.location_mask,
            blend_src_mask: &mut self.blend_src_mask,
            built_ins: &mut result_built_ins,
            capabilities: self.capabilities,
            flags: self.flags,
            mesh_output_type,
            has_task_payload: ep.task_payload.is_some(),
        };
        ctx.validate(ep, ty, None)
            .map_err_inner(|e| EntryPointError::Result(e).with_span())?;
        if mesh_output_type == MeshOutputType::PrimitiveOutput {
            let mut num_indices_builtins = 0;
            if result_built_ins.contains(&crate::BuiltIn::PointIndex) {
                num_indices_builtins += 1;
            }
            if result_built_ins.contains(&crate::BuiltIn::LineIndices) {
                num_indices_builtins += 1;
            }
            if result_built_ins.contains(&crate::BuiltIn::TriangleIndices) {
                num_indices_builtins += 1;
            }
            if num_indices_builtins != 1 {
                return Err(EntryPointError::InvalidMeshPrimitiveOutputType
                    .with_span_handle(ty, &module.types));
            }
        } else if mesh_output_type == MeshOutputType::VertexOutput
            && !result_built_ins.contains(&crate::BuiltIn::Position { invariant: false })
        {
            return Err(
                EntryPointError::MissingVertexOutputPosition.with_span_handle(ty, &module.types)
            );
        }

        Ok(())
    }

    pub(super) fn validate_entry_point(
        &mut self,
        ep: &crate::EntryPoint,
        module: &crate::Module,
        mod_info: &ModuleInfo,
    ) -> Result<FunctionInfo, WithSpan<EntryPointError>> {
        if matches!(
            ep.stage,
            crate::ShaderStage::Task | crate::ShaderStage::Mesh
        ) && !self.capabilities.contains(Capabilities::MESH_SHADER)
        {
            return Err(EntryPointError::MeshShaderCapabilityDisabled.with_span());
        }
        if ep.early_depth_test.is_some() {
            let required = Capabilities::EARLY_DEPTH_TEST;
            if !self.capabilities.contains(required) {
                return Err(
                    EntryPointError::Result(VaryingError::UnsupportedCapability(required))
                        .with_span(),
                );
            }

            if ep.stage != crate::ShaderStage::Fragment {
                return Err(EntryPointError::UnexpectedEarlyDepthTest.with_span());
            }
        }

        if ep.stage.compute_like() {
            if ep
                .workgroup_size
                .iter()
                .any(|&s| s == 0 || s > MAX_WORKGROUP_SIZE)
            {
                return Err(EntryPointError::OutOfRangeWorkgroupSize.with_span());
            }
        } else if ep.workgroup_size != [0; 3] {
            return Err(EntryPointError::UnexpectedWorkgroupSize.with_span());
        }

        match (ep.stage, &ep.mesh_info) {
            (crate::ShaderStage::Mesh, &None) => {
                return Err(EntryPointError::ExpectedMeshShaderAttributes.with_span());
            }
            (_, &Some(_)) => {
                return Err(EntryPointError::UnexpectedMeshShaderAttributes.with_span());
            }
            (_, _) => {}
        }

        let mut info = self
            .validate_function(&ep.function, module, mod_info, true)
            .map_err(WithSpan::into_other)?;

        // Validate the task shader payload.
        match ep.stage {
            // Task shaders must produce a payload.
            crate::ShaderStage::Task => {
                let Some(handle) = ep.task_payload else {
                    return Err(EntryPointError::ExpectedTaskPayload.with_span());
                };
                if module.global_variables[handle].space != crate::AddressSpace::TaskPayload {
                    return Err(EntryPointError::TaskPayloadWrongAddressSpace
                        .with_span_handle(handle, &module.global_variables));
                }
                info.insert_global_use(GlobalUse::READ | GlobalUse::WRITE, handle);
            }

            // Mesh shaders may accept a payload.
            crate::ShaderStage::Mesh => {
                if let Some(handle) = ep.task_payload {
                    if module.global_variables[handle].space != crate::AddressSpace::TaskPayload {
                        return Err(EntryPointError::TaskPayloadWrongAddressSpace
                            .with_span_handle(handle, &module.global_variables));
                    }
                    info.insert_global_use(GlobalUse::READ, handle);
                }
            }

            // Other stages must not have a payload.
            _ => {
                if let Some(handle) = ep.task_payload {
                    return Err(EntryPointError::UnexpectedTaskPayload
                        .with_span_handle(handle, &module.global_variables));
                }
            }
        }

        {
            use super::ShaderStages;

            let stage_bit = match ep.stage {
                crate::ShaderStage::Vertex => ShaderStages::VERTEX,
                crate::ShaderStage::Fragment => ShaderStages::FRAGMENT,
                crate::ShaderStage::Compute => ShaderStages::COMPUTE,
                crate::ShaderStage::Mesh => ShaderStages::MESH,
                crate::ShaderStage::Task => ShaderStages::TASK,
            };

            if !info.available_stages.contains(stage_bit) {
                return Err(EntryPointError::ForbiddenStageOperations.with_span());
            }
        }

        self.location_mask.clear();
        let mut argument_built_ins = crate::FastHashSet::default();
        // TODO: add span info to function arguments
        for (index, fa) in ep.function.arguments.iter().enumerate() {
            let mut ctx = VaryingContext {
                stage: ep.stage,
                output: false,
                types: &module.types,
                type_info: &self.types,
                location_mask: &mut self.location_mask,
                blend_src_mask: &mut self.blend_src_mask,
                built_ins: &mut argument_built_ins,
                capabilities: self.capabilities,
                flags: self.flags,
                mesh_output_type: MeshOutputType::None,
                has_task_payload: ep.task_payload.is_some(),
            };
            ctx.validate(ep, fa.ty, fa.binding.as_ref())
                .map_err_inner(|e| EntryPointError::Argument(index as u32, e).with_span())?;
        }

        self.location_mask.clear();
        if let Some(ref fr) = ep.function.result {
            let mut result_built_ins = crate::FastHashSet::default();
            let mut ctx = VaryingContext {
                stage: ep.stage,
                output: true,
                types: &module.types,
                type_info: &self.types,
                location_mask: &mut self.location_mask,
                blend_src_mask: &mut self.blend_src_mask,
                built_ins: &mut result_built_ins,
                capabilities: self.capabilities,
                flags: self.flags,
                mesh_output_type: MeshOutputType::None,
                has_task_payload: ep.task_payload.is_some(),
            };
            ctx.validate(ep, fr.ty, fr.binding.as_ref())
                .map_err_inner(|e| EntryPointError::Result(e).with_span())?;
            if ep.stage == crate::ShaderStage::Vertex
                && !result_built_ins.contains(&crate::BuiltIn::Position { invariant: false })
            {
                return Err(EntryPointError::MissingVertexOutputPosition.with_span());
            }
            if ep.stage == crate::ShaderStage::Mesh {
                return Err(EntryPointError::UnexpectedMeshShaderEntryResult.with_span());
            }
            // Task shaders must have a single `MeshTaskSize` output, and nothing else.
            if ep.stage == crate::ShaderStage::Task {
                let ok = result_built_ins.contains(&crate::BuiltIn::MeshTaskSize)
                    && result_built_ins.len() == 1
                    && self.location_mask.is_empty();
                if !ok {
                    return Err(EntryPointError::WrongTaskShaderEntryResult.with_span());
                }
            }
            if !self.blend_src_mask.is_empty() {
                info.dual_source_blending = true;
            }
        } else if ep.stage == crate::ShaderStage::Vertex {
            return Err(EntryPointError::MissingVertexOutputPosition.with_span());
        } else if ep.stage == crate::ShaderStage::Task {
            return Err(EntryPointError::WrongTaskShaderEntryResult.with_span());
        }

        {
            let mut used_push_constants = module
                .global_variables
                .iter()
                .filter(|&(_, var)| var.space == crate::AddressSpace::PushConstant)
                .map(|(handle, _)| handle)
                .filter(|&handle| !info[handle].is_empty());
            // Check if there is more than one push constant, and error if so.
            // Use a loop for when returning multiple errors is supported.
            if let Some(handle) = used_push_constants.nth(1) {
                return Err(EntryPointError::MoreThanOnePushConstantUsed
                    .with_span_handle(handle, &module.global_variables));
            }
        }

        self.ep_resource_bindings.clear();
        for (var_handle, var) in module.global_variables.iter() {
            let usage = info[var_handle];
            if usage.is_empty() {
                continue;
            }

            if var.space == crate::AddressSpace::TaskPayload {
                if ep.task_payload != Some(var_handle) {
                    return Err(EntryPointError::WrongTaskPayloadUsed
                        .with_span_handle(var_handle, &module.global_variables));
                }
            }

            let allowed_usage = match var.space {
                crate::AddressSpace::Function => unreachable!(),
                crate::AddressSpace::Uniform => GlobalUse::READ | GlobalUse::QUERY,
                crate::AddressSpace::Storage { access } => storage_usage(access),
                crate::AddressSpace::Handle => match module.types[var.ty].inner {
                    crate::TypeInner::BindingArray { base, .. } => match module.types[base].inner {
                        crate::TypeInner::Image {
                            class: crate::ImageClass::Storage { access, .. },
                            ..
                        } => storage_usage(access),
                        _ => GlobalUse::READ | GlobalUse::QUERY,
                    },
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage { access, .. },
                        ..
                    } => storage_usage(access),
                    _ => GlobalUse::READ | GlobalUse::QUERY,
                },
                crate::AddressSpace::Private | crate::AddressSpace::WorkGroup => {
                    GlobalUse::READ | GlobalUse::WRITE | GlobalUse::QUERY
                }
                crate::AddressSpace::TaskPayload => {
                    GlobalUse::READ
                        | GlobalUse::QUERY
                        | if ep.stage == crate::ShaderStage::Task {
                            GlobalUse::WRITE
                        } else {
                            GlobalUse::empty()
                        }
                }
                crate::AddressSpace::PushConstant => GlobalUse::READ,
            };
            if !allowed_usage.contains(usage) {
                log::warn!("\tUsage error for: {var:?}");
                log::warn!("\tAllowed usage: {allowed_usage:?}, requested: {usage:?}");
                return Err(EntryPointError::InvalidGlobalUsage(var_handle, usage)
                    .with_span_handle(var_handle, &module.global_variables));
            }

            if let Some(ref bind) = var.binding {
                if !self.ep_resource_bindings.insert(*bind) {
                    if self.flags.contains(super::ValidationFlags::BINDINGS) {
                        return Err(EntryPointError::BindingCollision(var_handle)
                            .with_span_handle(var_handle, &module.global_variables));
                    }
                }
            }
        }

        // If this is a `Mesh` entry point, check its vertex and primitive output types.
        // We verified previously that only mesh shaders can have `mesh_info`.
        if let &Some(ref mesh_info) = &ep.mesh_info {
            // Mesh shaders don't return any value. All their results are supplied through
            // [`SetVertex`] and [`SetPrimitive`] calls.
            if let Some((used_vertex_type, _)) = info.mesh_shader_info.vertex_type {
                if used_vertex_type != mesh_info.vertex_output_type {
                    return Err(EntryPointError::WrongMeshOutputType
                        .with_span_handle(mesh_info.vertex_output_type, &module.types));
                }
            }
            if let Some((used_primitive_type, _)) = info.mesh_shader_info.primitive_type {
                if used_primitive_type != mesh_info.primitive_output_type {
                    return Err(EntryPointError::WrongMeshOutputType
                        .with_span_handle(mesh_info.primitive_output_type, &module.types));
                }
            }

            self.validate_mesh_output_type(
                ep,
                module,
                mesh_info.vertex_output_type,
                MeshOutputType::VertexOutput,
            )?;
            self.validate_mesh_output_type(
                ep,
                module,
                mesh_info.primitive_output_type,
                MeshOutputType::PrimitiveOutput,
            )?;
        } else {
            // This is not a `Mesh` entry point, so ensure that it never tries to produce
            // vertices or primitives.
            if info.mesh_shader_info.vertex_type.is_some()
                || info.mesh_shader_info.primitive_type.is_some()
            {
                return Err(EntryPointError::UnexpectedMeshShaderOutput.with_span());
            }
        }

        Ok(info)
    }
}
