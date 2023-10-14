#[cfg(feature = "validate")]
use super::{
    compose::validate_compose, validate_atomic_compare_exchange_struct, FunctionInfo, ModuleInfo,
    ShaderStages, TypeFlags,
};
#[cfg(feature = "validate")]
use crate::arena::UniqueArena;

use crate::{
    arena::Handle,
    proc::{IndexableLengthError, ResolveError},
};

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ExpressionError {
    #[error("Doesn't exist")]
    DoesntExist,
    #[error("Used by a statement before it was introduced into the scope by any of the dominating blocks")]
    NotInScope,
    #[error("Base type {0:?} is not compatible with this expression")]
    InvalidBaseType(Handle<crate::Expression>),
    #[error("Accessing with index {0:?} can't be done")]
    InvalidIndexType(Handle<crate::Expression>),
    #[error("Accessing {0:?} via a negative index is invalid")]
    NegativeIndex(Handle<crate::Expression>),
    #[error("Accessing index {1} is out of {0:?} bounds")]
    IndexOutOfBounds(Handle<crate::Expression>, u32),
    #[error("The expression {0:?} may only be indexed by a constant")]
    IndexMustBeConstant(Handle<crate::Expression>),
    #[error("Function argument {0:?} doesn't exist")]
    FunctionArgumentDoesntExist(u32),
    #[error("Loading of {0:?} can't be done")]
    InvalidPointerType(Handle<crate::Expression>),
    #[error("Array length of {0:?} can't be done")]
    InvalidArrayType(Handle<crate::Expression>),
    #[error("Get intersection of {0:?} can't be done")]
    InvalidRayQueryType(Handle<crate::Expression>),
    #[error("Splatting {0:?} can't be done")]
    InvalidSplatType(Handle<crate::Expression>),
    #[error("Swizzling {0:?} can't be done")]
    InvalidVectorType(Handle<crate::Expression>),
    #[error("Swizzle component {0:?} is outside of vector size {1:?}")]
    InvalidSwizzleComponent(crate::SwizzleComponent, crate::VectorSize),
    #[error(transparent)]
    Compose(#[from] super::ComposeError),
    #[error(transparent)]
    IndexableLength(#[from] IndexableLengthError),
    #[error("Operation {0:?} can't work with {1:?}")]
    InvalidUnaryOperandType(crate::UnaryOperator, Handle<crate::Expression>),
    #[error("Operation {0:?} can't work with {1:?} and {2:?}")]
    InvalidBinaryOperandTypes(
        crate::BinaryOperator,
        Handle<crate::Expression>,
        Handle<crate::Expression>,
    ),
    #[error("Selecting is not possible")]
    InvalidSelectTypes,
    #[error("Relational argument {0:?} is not a boolean vector")]
    InvalidBooleanVector(Handle<crate::Expression>),
    #[error("Relational argument {0:?} is not a float")]
    InvalidFloatArgument(Handle<crate::Expression>),
    #[error("Type resolution failed")]
    Type(#[from] ResolveError),
    #[error("Not a global variable")]
    ExpectedGlobalVariable,
    #[error("Not a global variable or a function argument")]
    ExpectedGlobalOrArgument,
    #[error("Needs to be an binding array instead of {0:?}")]
    ExpectedBindingArrayType(Handle<crate::Type>),
    #[error("Needs to be an image instead of {0:?}")]
    ExpectedImageType(Handle<crate::Type>),
    #[error("Needs to be an image instead of {0:?}")]
    ExpectedSamplerType(Handle<crate::Type>),
    #[error("Unable to operate on image class {0:?}")]
    InvalidImageClass(crate::ImageClass),
    #[error("Derivatives can only be taken from scalar and vector floats")]
    InvalidDerivative,
    #[error("Image array index parameter is misplaced")]
    InvalidImageArrayIndex,
    #[error("Inappropriate sample or level-of-detail index for texel access")]
    InvalidImageOtherIndex,
    #[error("Image array index type of {0:?} is not an integer scalar")]
    InvalidImageArrayIndexType(Handle<crate::Expression>),
    #[error("Image sample or level-of-detail index's type of {0:?} is not an integer scalar")]
    InvalidImageOtherIndexType(Handle<crate::Expression>),
    #[error("Image coordinate type of {1:?} does not match dimension {0:?}")]
    InvalidImageCoordinateType(crate::ImageDimension, Handle<crate::Expression>),
    #[error("Comparison sampling mismatch: image has class {image:?}, but the sampler is comparison={sampler}, and the reference was provided={has_ref}")]
    ComparisonSamplingMismatch {
        image: crate::ImageClass,
        sampler: bool,
        has_ref: bool,
    },
    #[error("Sample offset constant {1:?} doesn't match the image dimension {0:?}")]
    InvalidSampleOffset(crate::ImageDimension, Handle<crate::Expression>),
    #[error("Depth reference {0:?} is not a scalar float")]
    InvalidDepthReference(Handle<crate::Expression>),
    #[error("Depth sample level can only be Auto or Zero")]
    InvalidDepthSampleLevel,
    #[error("Gather level can only be Zero")]
    InvalidGatherLevel,
    #[error("Gather component {0:?} doesn't exist in the image")]
    InvalidGatherComponent(crate::SwizzleComponent),
    #[error("Gather can't be done for image dimension {0:?}")]
    InvalidGatherDimension(crate::ImageDimension),
    #[error("Sample level (exact) type {0:?} is not a scalar float")]
    InvalidSampleLevelExactType(Handle<crate::Expression>),
    #[error("Sample level (bias) type {0:?} is not a scalar float")]
    InvalidSampleLevelBiasType(Handle<crate::Expression>),
    #[error("Sample level (gradient) of {1:?} doesn't match the image dimension {0:?}")]
    InvalidSampleLevelGradientType(crate::ImageDimension, Handle<crate::Expression>),
    #[error("Unable to cast")]
    InvalidCastArgument,
    #[error("Invalid argument count for {0:?}")]
    WrongArgumentCount(crate::MathFunction),
    #[error("Argument [{1}] to {0:?} as expression {2:?} has an invalid type.")]
    InvalidArgumentType(crate::MathFunction, u32, Handle<crate::Expression>),
    #[error("Atomic result type can't be {0:?}")]
    InvalidAtomicResultType(Handle<crate::Type>),
    #[error(
        "workgroupUniformLoad result type can't be {0:?}. It can only be a constructible type."
    )]
    InvalidWorkGroupUniformLoadResultType(Handle<crate::Type>),
    #[error("Shader requires capability {0:?}")]
    MissingCapabilities(super::Capabilities),
    #[error(transparent)]
    Literal(#[from] LiteralError),
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ConstExpressionError {
    #[error("The expression is not a constant expression")]
    NonConst,
    #[error(transparent)]
    Compose(#[from] super::ComposeError),
    #[error("Splatting {0:?} can't be done")]
    InvalidSplatType(Handle<crate::Expression>),
    #[error("Type resolution failed")]
    Type(#[from] ResolveError),
    #[error(transparent)]
    Literal(#[from] LiteralError),
    #[error(transparent)]
    Width(#[from] super::r#type::WidthError),
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum LiteralError {
    #[error("Float literal is NaN")]
    NaN,
    #[error("Float literal is infinite")]
    Infinity,
    #[error(transparent)]
    Width(#[from] super::r#type::WidthError),
}

#[cfg(feature = "validate")]
struct ExpressionTypeResolver<'a> {
    root: Handle<crate::Expression>,
    types: &'a UniqueArena<crate::Type>,
    info: &'a FunctionInfo,
}

#[cfg(feature = "validate")]
impl<'a> std::ops::Index<Handle<crate::Expression>> for ExpressionTypeResolver<'a> {
    type Output = crate::TypeInner;

    #[allow(clippy::panic)]
    fn index(&self, handle: Handle<crate::Expression>) -> &Self::Output {
        if handle < self.root {
            self.info[handle].ty.inner_with(self.types)
        } else {
            // `Validator::validate_module_handles` should have caught this.
            panic!(
                "Depends on {:?}, which has not been processed yet",
                self.root
            )
        }
    }
}

#[cfg(feature = "validate")]
impl super::Validator {
    pub(super) fn validate_const_expression(
        &self,
        handle: Handle<crate::Expression>,
        gctx: crate::proc::GlobalCtx,
        mod_info: &ModuleInfo,
    ) -> Result<(), ConstExpressionError> {
        use crate::Expression as E;

        match gctx.const_expressions[handle] {
            E::Literal(literal) => {
                self.validate_literal(literal)?;
            }
            E::Constant(_) | E::ZeroValue(_) => {}
            E::Compose { ref components, ty } => {
                validate_compose(
                    ty,
                    gctx,
                    components.iter().map(|&handle| mod_info[handle].clone()),
                )?;
            }
            E::Splat { value, .. } => match *mod_info[value].inner_with(gctx.types) {
                crate::TypeInner::Scalar { .. } => {}
                _ => return Err(super::ConstExpressionError::InvalidSplatType(value)),
            },
            _ => return Err(super::ConstExpressionError::NonConst),
        }

        Ok(())
    }

    pub(super) fn validate_expression(
        &self,
        root: Handle<crate::Expression>,
        expression: &crate::Expression,
        function: &crate::Function,
        module: &crate::Module,
        info: &FunctionInfo,
        mod_info: &ModuleInfo,
    ) -> Result<ShaderStages, ExpressionError> {
        use crate::{Expression as E, ScalarKind as Sk, TypeInner as Ti};

        let resolver = ExpressionTypeResolver {
            root,
            types: &module.types,
            info,
        };

        let stages = match *expression {
            E::Access { base, index } => {
                let base_type = &resolver[base];
                // See the documentation for `Expression::Access`.
                let dynamic_indexing_restricted = match *base_type {
                    Ti::Vector { .. } => false,
                    Ti::Matrix { .. } | Ti::Array { .. } => true,
                    Ti::Pointer { .. }
                    | Ti::ValuePointer { size: Some(_), .. }
                    | Ti::BindingArray { .. } => false,
                    ref other => {
                        log::error!("Indexing of {:?}", other);
                        return Err(ExpressionError::InvalidBaseType(base));
                    }
                };
                match resolver[index] {
                    //TODO: only allow one of these
                    Ti::Scalar {
                        kind: Sk::Sint | Sk::Uint,
                        width: _,
                    } => {}
                    ref other => {
                        log::error!("Indexing by {:?}", other);
                        return Err(ExpressionError::InvalidIndexType(index));
                    }
                }
                if dynamic_indexing_restricted
                    && function.expressions[index].is_dynamic_index(module)
                {
                    return Err(ExpressionError::IndexMustBeConstant(base));
                }

                // If we know both the length and the index, we can do the
                // bounds check now.
                if let crate::proc::IndexableLength::Known(known_length) =
                    base_type.indexable_length(module)?
                {
                    match module
                        .to_ctx()
                        .eval_expr_to_u32_from(index, &function.expressions)
                    {
                        Ok(value) => {
                            if value >= known_length {
                                return Err(ExpressionError::IndexOutOfBounds(base, value));
                            }
                        }
                        Err(crate::proc::U32EvalError::Negative) => {
                            return Err(ExpressionError::NegativeIndex(base))
                        }
                        Err(crate::proc::U32EvalError::NonConst) => {}
                    }
                }

                ShaderStages::all()
            }
            E::AccessIndex { base, index } => {
                fn resolve_index_limit(
                    module: &crate::Module,
                    top: Handle<crate::Expression>,
                    ty: &crate::TypeInner,
                    top_level: bool,
                ) -> Result<u32, ExpressionError> {
                    let limit = match *ty {
                        Ti::Vector { size, .. }
                        | Ti::ValuePointer {
                            size: Some(size), ..
                        } => size as u32,
                        Ti::Matrix { columns, .. } => columns as u32,
                        Ti::Array {
                            size: crate::ArraySize::Constant(len),
                            ..
                        } => len.get(),
                        Ti::Array { .. } | Ti::BindingArray { .. } => u32::MAX, // can't statically know, but need run-time checks
                        Ti::Pointer { base, .. } if top_level => {
                            resolve_index_limit(module, top, &module.types[base].inner, false)?
                        }
                        Ti::Struct { ref members, .. } => members.len() as u32,
                        ref other => {
                            log::error!("Indexing of {:?}", other);
                            return Err(ExpressionError::InvalidBaseType(top));
                        }
                    };
                    Ok(limit)
                }

                let limit = resolve_index_limit(module, base, &resolver[base], true)?;
                if index >= limit {
                    return Err(ExpressionError::IndexOutOfBounds(base, limit));
                }
                ShaderStages::all()
            }
            E::Splat { size: _, value } => match resolver[value] {
                Ti::Scalar { .. } => ShaderStages::all(),
                ref other => {
                    log::error!("Splat scalar type {:?}", other);
                    return Err(ExpressionError::InvalidSplatType(value));
                }
            },
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let vec_size = match resolver[vector] {
                    Ti::Vector { size: vec_size, .. } => vec_size,
                    ref other => {
                        log::error!("Swizzle vector type {:?}", other);
                        return Err(ExpressionError::InvalidVectorType(vector));
                    }
                };
                for &sc in pattern[..size as usize].iter() {
                    if sc as u8 >= vec_size as u8 {
                        return Err(ExpressionError::InvalidSwizzleComponent(sc, vec_size));
                    }
                }
                ShaderStages::all()
            }
            E::Literal(literal) => {
                self.validate_literal(literal)?;
                ShaderStages::all()
            }
            E::Constant(_) | E::ZeroValue(_) => ShaderStages::all(),
            E::Compose { ref components, ty } => {
                validate_compose(
                    ty,
                    module.to_ctx(),
                    components.iter().map(|&handle| info[handle].ty.clone()),
                )?;
                ShaderStages::all()
            }
            E::FunctionArgument(index) => {
                if index >= function.arguments.len() as u32 {
                    return Err(ExpressionError::FunctionArgumentDoesntExist(index));
                }
                ShaderStages::all()
            }
            E::GlobalVariable(_handle) => ShaderStages::all(),
            E::LocalVariable(_handle) => ShaderStages::all(),
            E::Load { pointer } => {
                match resolver[pointer] {
                    Ti::Pointer { base, .. }
                        if self.types[base.index()]
                            .flags
                            .contains(TypeFlags::SIZED | TypeFlags::DATA) => {}
                    Ti::ValuePointer { .. } => {}
                    ref other => {
                        log::error!("Loading {:?}", other);
                        return Err(ExpressionError::InvalidPointerType(pointer));
                    }
                }
                ShaderStages::all()
            }
            E::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                // check the validity of expressions
                let image_ty = Self::global_var_ty(module, function, image)?;
                let sampler_ty = Self::global_var_ty(module, function, sampler)?;

                let comparison = match module.types[sampler_ty].inner {
                    Ti::Sampler { comparison } => comparison,
                    _ => return Err(ExpressionError::ExpectedSamplerType(sampler_ty)),
                };

                let (class, dim) = match module.types[image_ty].inner {
                    Ti::Image {
                        class,
                        arrayed,
                        dim,
                    } => {
                        // check the array property
                        if arrayed != array_index.is_some() {
                            return Err(ExpressionError::InvalidImageArrayIndex);
                        }
                        if let Some(expr) = array_index {
                            match resolver[expr] {
                                Ti::Scalar {
                                    kind: Sk::Sint | Sk::Uint,
                                    width: _,
                                } => {}
                                _ => return Err(ExpressionError::InvalidImageArrayIndexType(expr)),
                            }
                        }
                        (class, dim)
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(image_ty)),
                };

                // check sampling and comparison properties
                let image_depth = match class {
                    crate::ImageClass::Sampled {
                        kind: crate::ScalarKind::Float,
                        multi: false,
                    } => false,
                    crate::ImageClass::Sampled {
                        kind: crate::ScalarKind::Uint | crate::ScalarKind::Sint,
                        multi: false,
                    } if gather.is_some() => false,
                    crate::ImageClass::Depth { multi: false } => true,
                    _ => return Err(ExpressionError::InvalidImageClass(class)),
                };
                if comparison != depth_ref.is_some() || (comparison && !image_depth) {
                    return Err(ExpressionError::ComparisonSamplingMismatch {
                        image: class,
                        sampler: comparison,
                        has_ref: depth_ref.is_some(),
                    });
                }

                // check texture coordinates type
                let num_components = match dim {
                    crate::ImageDimension::D1 => 1,
                    crate::ImageDimension::D2 => 2,
                    crate::ImageDimension::D3 | crate::ImageDimension::Cube => 3,
                };
                match resolver[coordinate] {
                    Ti::Scalar {
                        kind: Sk::Float, ..
                    } if num_components == 1 => {}
                    Ti::Vector {
                        size,
                        kind: Sk::Float,
                        ..
                    } if size as u32 == num_components => {}
                    _ => return Err(ExpressionError::InvalidImageCoordinateType(dim, coordinate)),
                }

                // check constant offset
                if let Some(const_expr) = offset {
                    match *mod_info[const_expr].inner_with(&module.types) {
                        Ti::Scalar { kind: Sk::Sint, .. } if num_components == 1 => {}
                        Ti::Vector {
                            size,
                            kind: Sk::Sint,
                            ..
                        } if size as u32 == num_components => {}
                        _ => {
                            return Err(ExpressionError::InvalidSampleOffset(dim, const_expr));
                        }
                    }
                }

                // check depth reference type
                if let Some(expr) = depth_ref {
                    match resolver[expr] {
                        Ti::Scalar {
                            kind: Sk::Float, ..
                        } => {}
                        _ => return Err(ExpressionError::InvalidDepthReference(expr)),
                    }
                    match level {
                        crate::SampleLevel::Auto | crate::SampleLevel::Zero => {}
                        _ => return Err(ExpressionError::InvalidDepthSampleLevel),
                    }
                }

                if let Some(component) = gather {
                    match dim {
                        crate::ImageDimension::D2 | crate::ImageDimension::Cube => {}
                        crate::ImageDimension::D1 | crate::ImageDimension::D3 => {
                            return Err(ExpressionError::InvalidGatherDimension(dim))
                        }
                    };
                    let max_component = match class {
                        crate::ImageClass::Depth { .. } => crate::SwizzleComponent::X,
                        _ => crate::SwizzleComponent::W,
                    };
                    if component > max_component {
                        return Err(ExpressionError::InvalidGatherComponent(component));
                    }
                    match level {
                        crate::SampleLevel::Zero => {}
                        _ => return Err(ExpressionError::InvalidGatherLevel),
                    }
                }

                // check level properties
                match level {
                    crate::SampleLevel::Auto => ShaderStages::FRAGMENT,
                    crate::SampleLevel::Zero => ShaderStages::all(),
                    crate::SampleLevel::Exact(expr) => {
                        match resolver[expr] {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidSampleLevelExactType(expr)),
                        }
                        ShaderStages::all()
                    }
                    crate::SampleLevel::Bias(expr) => {
                        match resolver[expr] {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidSampleLevelBiasType(expr)),
                        }
                        ShaderStages::FRAGMENT
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        match resolver[x] {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } if num_components == 1 => {}
                            Ti::Vector {
                                size,
                                kind: Sk::Float,
                                ..
                            } if size as u32 == num_components => {}
                            _ => {
                                return Err(ExpressionError::InvalidSampleLevelGradientType(dim, x))
                            }
                        }
                        match resolver[y] {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } if num_components == 1 => {}
                            Ti::Vector {
                                size,
                                kind: Sk::Float,
                                ..
                            } if size as u32 == num_components => {}
                            _ => {
                                return Err(ExpressionError::InvalidSampleLevelGradientType(dim, y))
                            }
                        }
                        ShaderStages::all()
                    }
                }
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let ty = Self::global_var_ty(module, function, image)?;
                match module.types[ty].inner {
                    Ti::Image {
                        class,
                        arrayed,
                        dim,
                    } => {
                        match resolver[coordinate].image_storage_coordinates() {
                            Some(coord_dim) if coord_dim == dim => {}
                            _ => {
                                return Err(ExpressionError::InvalidImageCoordinateType(
                                    dim, coordinate,
                                ))
                            }
                        };
                        if arrayed != array_index.is_some() {
                            return Err(ExpressionError::InvalidImageArrayIndex);
                        }
                        if let Some(expr) = array_index {
                            match resolver[expr] {
                                Ti::Scalar {
                                    kind: Sk::Sint | Sk::Uint,
                                    width: _,
                                } => {}
                                _ => return Err(ExpressionError::InvalidImageArrayIndexType(expr)),
                            }
                        }

                        match (sample, class.is_multisampled()) {
                            (None, false) => {}
                            (Some(sample), true) => {
                                if resolver[sample].scalar_kind() != Some(Sk::Sint) {
                                    return Err(ExpressionError::InvalidImageOtherIndexType(
                                        sample,
                                    ));
                                }
                            }
                            _ => {
                                return Err(ExpressionError::InvalidImageOtherIndex);
                            }
                        }

                        match (level, class.is_mipmapped()) {
                            (None, false) => {}
                            (Some(level), true) => {
                                if resolver[level].scalar_kind() != Some(Sk::Sint) {
                                    return Err(ExpressionError::InvalidImageOtherIndexType(level));
                                }
                            }
                            _ => {
                                return Err(ExpressionError::InvalidImageOtherIndex);
                            }
                        }
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(ty)),
                }
                ShaderStages::all()
            }
            E::ImageQuery { image, query } => {
                let ty = Self::global_var_ty(module, function, image)?;
                match module.types[ty].inner {
                    Ti::Image { class, arrayed, .. } => {
                        let good = match query {
                            crate::ImageQuery::NumLayers => arrayed,
                            crate::ImageQuery::Size { level: None } => true,
                            crate::ImageQuery::Size { level: Some(_) }
                            | crate::ImageQuery::NumLevels => class.is_mipmapped(),
                            crate::ImageQuery::NumSamples => class.is_multisampled(),
                        };
                        if !good {
                            return Err(ExpressionError::InvalidImageClass(class));
                        }
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(ty)),
                }
                ShaderStages::all()
            }
            E::Unary { op, expr } => {
                use crate::UnaryOperator as Uo;
                let inner = &resolver[expr];
                match (op, inner.scalar_kind()) {
                    (Uo::Negate, Some(Sk::Float | Sk::Sint))
                    | (Uo::LogicalNot, Some(Sk::Bool))
                    | (Uo::BitwiseNot, Some(Sk::Sint | Sk::Uint)) => {}
                    other => {
                        log::error!("Op {:?} kind {:?}", op, other);
                        return Err(ExpressionError::InvalidUnaryOperandType(op, expr));
                    }
                }
                ShaderStages::all()
            }
            E::Binary { op, left, right } => {
                use crate::BinaryOperator as Bo;
                let left_inner = &resolver[left];
                let right_inner = &resolver[right];
                let good = match op {
                    Bo::Add | Bo::Subtract => match *left_inner {
                        Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                            Sk::Uint | Sk::Sint | Sk::Float => left_inner == right_inner,
                            Sk::Bool => false,
                        },
                        Ti::Matrix { .. } => left_inner == right_inner,
                        _ => false,
                    },
                    Bo::Divide | Bo::Modulo => match *left_inner {
                        Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                            Sk::Uint | Sk::Sint | Sk::Float => left_inner == right_inner,
                            Sk::Bool => false,
                        },
                        _ => false,
                    },
                    Bo::Multiply => {
                        let kind_allowed = match left_inner.scalar_kind() {
                            Some(Sk::Uint | Sk::Sint | Sk::Float) => true,
                            Some(Sk::Bool) | None => false,
                        };
                        let types_match = match (left_inner, right_inner) {
                            // Straight scalar and mixed scalar/vector.
                            (&Ti::Scalar { kind: kind1, .. }, &Ti::Scalar { kind: kind2, .. })
                            | (&Ti::Vector { kind: kind1, .. }, &Ti::Scalar { kind: kind2, .. })
                            | (&Ti::Scalar { kind: kind1, .. }, &Ti::Vector { kind: kind2, .. }) => {
                                kind1 == kind2
                            }
                            // Scalar/matrix.
                            (
                                &Ti::Scalar {
                                    kind: Sk::Float, ..
                                },
                                &Ti::Matrix { .. },
                            )
                            | (
                                &Ti::Matrix { .. },
                                &Ti::Scalar {
                                    kind: Sk::Float, ..
                                },
                            ) => true,
                            // Vector/vector.
                            (
                                &Ti::Vector {
                                    kind: kind1,
                                    size: size1,
                                    ..
                                },
                                &Ti::Vector {
                                    kind: kind2,
                                    size: size2,
                                    ..
                                },
                            ) => kind1 == kind2 && size1 == size2,
                            // Matrix * vector.
                            (
                                &Ti::Matrix { columns, .. },
                                &Ti::Vector {
                                    kind: Sk::Float,
                                    size,
                                    ..
                                },
                            ) => columns == size,
                            // Vector * matrix.
                            (
                                &Ti::Vector {
                                    kind: Sk::Float,
                                    size,
                                    ..
                                },
                                &Ti::Matrix { rows, .. },
                            ) => size == rows,
                            (&Ti::Matrix { columns, .. }, &Ti::Matrix { rows, .. }) => {
                                columns == rows
                            }
                            _ => false,
                        };
                        let left_width = match *left_inner {
                            Ti::Scalar { width, .. }
                            | Ti::Vector { width, .. }
                            | Ti::Matrix { width, .. } => width,
                            _ => 0,
                        };
                        let right_width = match *right_inner {
                            Ti::Scalar { width, .. }
                            | Ti::Vector { width, .. }
                            | Ti::Matrix { width, .. } => width,
                            _ => 0,
                        };
                        kind_allowed && types_match && left_width == right_width
                    }
                    Bo::Equal | Bo::NotEqual => left_inner.is_sized() && left_inner == right_inner,
                    Bo::Less | Bo::LessEqual | Bo::Greater | Bo::GreaterEqual => {
                        match *left_inner {
                            Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                                Sk::Uint | Sk::Sint | Sk::Float => left_inner == right_inner,
                                Sk::Bool => false,
                            },
                            ref other => {
                                log::error!("Op {:?} left type {:?}", op, other);
                                false
                            }
                        }
                    }
                    Bo::LogicalAnd | Bo::LogicalOr => match *left_inner {
                        Ti::Scalar { kind: Sk::Bool, .. } | Ti::Vector { kind: Sk::Bool, .. } => {
                            left_inner == right_inner
                        }
                        ref other => {
                            log::error!("Op {:?} left type {:?}", op, other);
                            false
                        }
                    },
                    Bo::And | Bo::InclusiveOr => match *left_inner {
                        Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                            Sk::Bool | Sk::Sint | Sk::Uint => left_inner == right_inner,
                            Sk::Float => false,
                        },
                        ref other => {
                            log::error!("Op {:?} left type {:?}", op, other);
                            false
                        }
                    },
                    Bo::ExclusiveOr => match *left_inner {
                        Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                            Sk::Sint | Sk::Uint => left_inner == right_inner,
                            Sk::Bool | Sk::Float => false,
                        },
                        ref other => {
                            log::error!("Op {:?} left type {:?}", op, other);
                            false
                        }
                    },
                    Bo::ShiftLeft | Bo::ShiftRight => {
                        let (base_size, base_kind) = match *left_inner {
                            Ti::Scalar { kind, .. } => (Ok(None), kind),
                            Ti::Vector { size, kind, .. } => (Ok(Some(size)), kind),
                            ref other => {
                                log::error!("Op {:?} base type {:?}", op, other);
                                (Err(()), Sk::Bool)
                            }
                        };
                        let shift_size = match *right_inner {
                            Ti::Scalar { kind: Sk::Uint, .. } => Ok(None),
                            Ti::Vector {
                                size,
                                kind: Sk::Uint,
                                ..
                            } => Ok(Some(size)),
                            ref other => {
                                log::error!("Op {:?} shift type {:?}", op, other);
                                Err(())
                            }
                        };
                        match base_kind {
                            Sk::Sint | Sk::Uint => base_size.is_ok() && base_size == shift_size,
                            Sk::Float | Sk::Bool => false,
                        }
                    }
                };
                if !good {
                    log::error!(
                        "Left: {:?} of type {:?}",
                        function.expressions[left],
                        left_inner
                    );
                    log::error!(
                        "Right: {:?} of type {:?}",
                        function.expressions[right],
                        right_inner
                    );
                    return Err(ExpressionError::InvalidBinaryOperandTypes(op, left, right));
                }
                ShaderStages::all()
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let accept_inner = &resolver[accept];
                let reject_inner = &resolver[reject];
                let condition_good = match resolver[condition] {
                    Ti::Scalar {
                        kind: Sk::Bool,
                        width: _,
                    } => {
                        // When `condition` is a single boolean, `accept` and
                        // `reject` can be vectors or scalars.
                        match *accept_inner {
                            Ti::Scalar { .. } | Ti::Vector { .. } => true,
                            _ => false,
                        }
                    }
                    Ti::Vector {
                        size,
                        kind: Sk::Bool,
                        width: _,
                    } => match *accept_inner {
                        Ti::Vector {
                            size: other_size, ..
                        } => size == other_size,
                        _ => false,
                    },
                    _ => false,
                };
                if !condition_good || accept_inner != reject_inner {
                    return Err(ExpressionError::InvalidSelectTypes);
                }
                ShaderStages::all()
            }
            E::Derivative { expr, .. } => {
                match resolver[expr] {
                    Ti::Scalar {
                        kind: Sk::Float, ..
                    }
                    | Ti::Vector {
                        kind: Sk::Float, ..
                    } => {}
                    _ => return Err(ExpressionError::InvalidDerivative),
                }
                ShaderStages::FRAGMENT
            }
            E::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;
                let argument_inner = &resolver[argument];
                match fun {
                    Rf::All | Rf::Any => match *argument_inner {
                        Ti::Vector { kind: Sk::Bool, .. } => {}
                        ref other => {
                            log::error!("All/Any of type {:?}", other);
                            return Err(ExpressionError::InvalidBooleanVector(argument));
                        }
                    },
                    Rf::IsNan | Rf::IsInf => match *argument_inner {
                        Ti::Scalar {
                            kind: Sk::Float, ..
                        }
                        | Ti::Vector {
                            kind: Sk::Float, ..
                        } => {}
                        ref other => {
                            log::error!("Float test of type {:?}", other);
                            return Err(ExpressionError::InvalidFloatArgument(argument));
                        }
                    },
                }
                ShaderStages::all()
            }
            E::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                use crate::MathFunction as Mf;

                let resolve = |arg| &resolver[arg];
                let arg_ty = resolve(arg);
                let arg1_ty = arg1.map(resolve);
                let arg2_ty = arg2.map(resolve);
                let arg3_ty = arg3.map(resolve);
                match fun {
                    Mf::Abs => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        let good = match *arg_ty {
                            Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => kind != Sk::Bool,
                            _ => false,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidArgumentType(fun, 0, arg));
                        }
                    }
                    Mf::Min | Mf::Max => {
                        let arg1_ty = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), None, None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        let good = match *arg_ty {
                            Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => kind != Sk::Bool,
                            _ => false,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidArgumentType(fun, 0, arg));
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                    }
                    Mf::Clamp => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), None) => (ty1, ty2),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        let good = match *arg_ty {
                            Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => kind != Sk::Bool,
                            _ => false,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidArgumentType(fun, 0, arg));
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                        if arg2_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                2,
                                arg2.unwrap(),
                            ));
                        }
                    }
                    Mf::Saturate
                    | Mf::Cos
                    | Mf::Cosh
                    | Mf::Sin
                    | Mf::Sinh
                    | Mf::Tan
                    | Mf::Tanh
                    | Mf::Acos
                    | Mf::Asin
                    | Mf::Atan
                    | Mf::Asinh
                    | Mf::Acosh
                    | Mf::Atanh
                    | Mf::Radians
                    | Mf::Degrees
                    | Mf::Ceil
                    | Mf::Floor
                    | Mf::Round
                    | Mf::Fract
                    | Mf::Trunc
                    | Mf::Exp
                    | Mf::Exp2
                    | Mf::Log
                    | Mf::Log2
                    | Mf::Length
                    | Mf::Sqrt
                    | Mf::InverseSqrt => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            }
                            | Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::Sign => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float | Sk::Sint,
                                ..
                            }
                            | Ti::Vector {
                                kind: Sk::Float | Sk::Sint,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::Atan2 | Mf::Pow | Mf::Distance | Mf::Step => {
                        let arg1_ty = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), None, None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            }
                            | Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                    }
                    Mf::Modf | Mf::Frexp => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        if !matches!(
                            *arg_ty,
                            Ti::Scalar {
                                kind: Sk::Float,
                                ..
                            } | Ti::Vector {
                                kind: Sk::Float,
                                ..
                            },
                        ) {
                            return Err(ExpressionError::InvalidArgumentType(fun, 1, arg));
                        }
                    }
                    Mf::Ldexp => {
                        let arg1_ty = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), None, None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        let size0 = match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } => None,
                            Ti::Vector {
                                kind: Sk::Float,
                                size,
                                ..
                            } => Some(size),
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(fun, 0, arg));
                            }
                        };
                        let good = match *arg1_ty {
                            Ti::Scalar { kind: Sk::Sint, .. } if size0.is_none() => true,
                            Ti::Vector {
                                size,
                                kind: Sk::Sint,
                                ..
                            } if Some(size) == size0 => true,
                            _ => false,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                    }
                    Mf::Dot => {
                        let arg1_ty = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), None, None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Vector {
                                kind: Sk::Float | Sk::Sint | Sk::Uint,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                    }
                    Mf::Outer | Mf::Cross | Mf::Reflect => {
                        let arg1_ty = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), None, None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                    }
                    Mf::Refract => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), None) => (ty1, ty2),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };

                        match *arg_ty {
                            Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }

                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }

                        match (arg_ty, arg2_ty) {
                            (
                                &Ti::Vector {
                                    width: vector_width,
                                    ..
                                },
                                &Ti::Scalar {
                                    width: scalar_width,
                                    kind: Sk::Float,
                                },
                            ) if vector_width == scalar_width => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg2.unwrap(),
                                ))
                            }
                        }
                    }
                    Mf::Normalize => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::FaceForward | Mf::Fma | Mf::SmoothStep => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), None) => (ty1, ty2),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            }
                            | Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                        if arg2_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                2,
                                arg2.unwrap(),
                            ));
                        }
                    }
                    Mf::Mix => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), None) => (ty1, ty2),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        let arg_width = match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float,
                                width,
                            }
                            | Ti::Vector {
                                kind: Sk::Float,
                                width,
                                ..
                            } => width,
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        };
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                        // the last argument can always be a scalar
                        match *arg2_ty {
                            Ti::Scalar {
                                kind: Sk::Float,
                                width,
                            } if width == arg_width => {}
                            _ if arg2_ty == arg_ty => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg2.unwrap(),
                                ));
                            }
                        }
                    }
                    Mf::Inverse | Mf::Determinant => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        let good = match *arg_ty {
                            Ti::Matrix { columns, rows, .. } => columns == rows,
                            _ => false,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidArgumentType(fun, 0, arg));
                        }
                    }
                    Mf::Transpose => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Matrix { .. } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::CountTrailingZeros
                    | Mf::CountLeadingZeros
                    | Mf::CountOneBits
                    | Mf::ReverseBits
                    | Mf::FindLsb
                    | Mf::FindMsb => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            }
                            | Ti::Vector {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::InsertBits => {
                        let (arg1_ty, arg2_ty, arg3_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), Some(ty3)) => (ty1, ty2, ty3),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            }
                            | Ti::Vector {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        if arg1_ty != arg_ty {
                            return Err(ExpressionError::InvalidArgumentType(
                                fun,
                                1,
                                arg1.unwrap(),
                            ));
                        }
                        match *arg2_ty {
                            Ti::Scalar { kind: Sk::Uint, .. } => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg2.unwrap(),
                                ))
                            }
                        }
                        match *arg3_ty {
                            Ti::Scalar { kind: Sk::Uint, .. } => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg3.unwrap(),
                                ))
                            }
                        }
                    }
                    Mf::ExtractBits => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty, arg3_ty) {
                            (Some(ty1), Some(ty2), None) => (ty1, ty2),
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            }
                            | Ti::Vector {
                                kind: Sk::Sint | Sk::Uint,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                        match *arg1_ty {
                            Ti::Scalar { kind: Sk::Uint, .. } => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg1.unwrap(),
                                ))
                            }
                        }
                        match *arg2_ty {
                            Ti::Scalar { kind: Sk::Uint, .. } => {}
                            _ => {
                                return Err(ExpressionError::InvalidArgumentType(
                                    fun,
                                    2,
                                    arg2.unwrap(),
                                ))
                            }
                        }
                    }
                    Mf::Pack2x16unorm | Mf::Pack2x16snorm | Mf::Pack2x16float => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Vector {
                                size: crate::VectorSize::Bi,
                                kind: Sk::Float,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::Pack4x8snorm | Mf::Pack4x8unorm => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Vector {
                                size: crate::VectorSize::Quad,
                                kind: Sk::Float,
                                ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::Unpack2x16float
                    | Mf::Unpack2x16snorm
                    | Mf::Unpack2x16unorm
                    | Mf::Unpack4x8snorm
                    | Mf::Unpack4x8unorm => {
                        if arg1_ty.is_some() || arg2_ty.is_some() || arg3_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Scalar { kind: Sk::Uint, .. } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                }
                ShaderStages::all()
            }
            E::As {
                expr,
                kind,
                convert,
            } => {
                let base_width = match resolver[expr] {
                    crate::TypeInner::Scalar { width, .. }
                    | crate::TypeInner::Vector { width, .. }
                    | crate::TypeInner::Matrix { width, .. } => width,
                    _ => return Err(ExpressionError::InvalidCastArgument),
                };
                let width = convert.unwrap_or(base_width);
                if self.check_width(kind, width).is_err() {
                    return Err(ExpressionError::InvalidCastArgument);
                }
                ShaderStages::all()
            }
            E::CallResult(function) => mod_info.functions[function.index()].available_stages,
            E::AtomicResult { ty, comparison } => {
                let scalar_predicate = |ty: &crate::TypeInner| match ty {
                    &crate::TypeInner::Scalar {
                        kind: kind @ (crate::ScalarKind::Uint | crate::ScalarKind::Sint),
                        width,
                    } => self.check_width(kind, width).is_ok(),
                    _ => false,
                };
                let good = match &module.types[ty].inner {
                    ty if !comparison => scalar_predicate(ty),
                    &crate::TypeInner::Struct { ref members, .. } if comparison => {
                        validate_atomic_compare_exchange_struct(
                            &module.types,
                            members,
                            scalar_predicate,
                        )
                    }
                    _ => false,
                };
                if !good {
                    return Err(ExpressionError::InvalidAtomicResultType(ty));
                }
                ShaderStages::all()
            }
            E::WorkGroupUniformLoadResult { ty } => {
                if self.types[ty.index()]
                    .flags
                    // Sized | Constructible is exactly the types currently supported by
                    // WorkGroupUniformLoad
                    .contains(TypeFlags::SIZED | TypeFlags::CONSTRUCTIBLE)
                {
                    ShaderStages::COMPUTE
                } else {
                    return Err(ExpressionError::InvalidWorkGroupUniformLoadResultType(ty));
                }
            }
            E::ArrayLength(expr) => match resolver[expr] {
                Ti::Pointer { base, .. } => {
                    let base_ty = &resolver.types[base];
                    if let Ti::Array {
                        size: crate::ArraySize::Dynamic,
                        ..
                    } = base_ty.inner
                    {
                        ShaderStages::all()
                    } else {
                        return Err(ExpressionError::InvalidArrayType(expr));
                    }
                }
                ref other => {
                    log::error!("Array length of {:?}", other);
                    return Err(ExpressionError::InvalidArrayType(expr));
                }
            },
            E::RayQueryProceedResult => ShaderStages::all(),
            E::RayQueryGetIntersection {
                query,
                committed: _,
            } => match resolver[query] {
                Ti::Pointer {
                    base,
                    space: crate::AddressSpace::Function,
                } => match resolver.types[base].inner {
                    Ti::RayQuery => ShaderStages::all(),
                    ref other => {
                        log::error!("Intersection result of a pointer to {:?}", other);
                        return Err(ExpressionError::InvalidRayQueryType(query));
                    }
                },
                ref other => {
                    log::error!("Intersection result of {:?}", other);
                    return Err(ExpressionError::InvalidRayQueryType(query));
                }
            },
            E::SubgroupBallotResult => ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            E::SubgroupOperationResult { ty } => ShaderStages::COMPUTE, // FIXME
        };
        Ok(stages)
    }

    fn global_var_ty(
        module: &crate::Module,
        function: &crate::Function,
        expr: Handle<crate::Expression>,
    ) -> Result<Handle<crate::Type>, ExpressionError> {
        use crate::Expression as Ex;

        match function.expressions[expr] {
            Ex::GlobalVariable(var_handle) => Ok(module.global_variables[var_handle].ty),
            Ex::FunctionArgument(i) => Ok(function.arguments[i as usize].ty),
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                match function.expressions[base] {
                    Ex::GlobalVariable(var_handle) => {
                        let array_ty = module.global_variables[var_handle].ty;

                        match module.types[array_ty].inner {
                            crate::TypeInner::BindingArray { base, .. } => Ok(base),
                            _ => Err(ExpressionError::ExpectedBindingArrayType(array_ty)),
                        }
                    }
                    _ => Err(ExpressionError::ExpectedGlobalVariable),
                }
            }
            _ => Err(ExpressionError::ExpectedGlobalVariable),
        }
    }

    pub fn validate_literal(&self, literal: crate::Literal) -> Result<(), LiteralError> {
        let kind = literal.scalar_kind();
        let width = literal.width();
        self.check_width(kind, width)?;
        check_literal_value(literal)?;

        Ok(())
    }
}

pub fn check_literal_value(literal: crate::Literal) -> Result<(), LiteralError> {
    let is_nan = match literal {
        crate::Literal::F64(v) => v.is_nan(),
        crate::Literal::F32(v) => v.is_nan(),
        _ => false,
    };
    if is_nan {
        return Err(LiteralError::NaN);
    }

    let is_infinite = match literal {
        crate::Literal::F64(v) => v.is_infinite(),
        crate::Literal::F32(v) => v.is_infinite(),
        _ => false,
    };
    if is_infinite {
        return Err(LiteralError::Infinity);
    }

    Ok(())
}
