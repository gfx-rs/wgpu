use super::{compose::validate_compose, ComposeError, FunctionInfo, ShaderStages, TypeFlags};
use crate::{
    arena::{Arena, Handle},
    proc::ResolveError,
};

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ExpressionError {
    #[error("Doesn't exist")]
    DoesntExist,
    #[error("Used by a statement before it was introduced into the scope by any of the dominating blocks")]
    NotInScope,
    #[error("Depends on {0:?}, which has not been processed yet")]
    ForwardDependency(Handle<crate::Expression>),
    #[error("Base type {0:?} is not compatible with this expression")]
    InvalidBaseType(Handle<crate::Expression>),
    #[error("Accessing with index {0:?} can't be done")]
    InvalidIndexType(Handle<crate::Expression>),
    #[error("Accessing index {1} is out of {0:?} bounds")]
    IndexOutOfBounds(Handle<crate::Expression>, u32),
    #[error("Function argument {0:?} doesn't exist")]
    FunctionArgumentDoesntExist(u32),
    #[error("Constant {0:?} doesn't exist")]
    ConstantDoesntExist(Handle<crate::Constant>),
    #[error("Global variable {0:?} doesn't exist")]
    GlobalVarDoesntExist(Handle<crate::GlobalVariable>),
    #[error("Local variable {0:?} doesn't exist")]
    LocalVarDoesntExist(Handle<crate::LocalVariable>),
    #[error("Loading of {0:?} can't be done")]
    InvalidPointerType(Handle<crate::Expression>),
    #[error("Array length of {0:?} can't be done")]
    InvalidArrayType(Handle<crate::Expression>),
    #[error("Splatting {0:?} can't be done")]
    InvalidSplatType(Handle<crate::Expression>),
    #[error("Swizzling {0:?} can't be done")]
    InvalidVectorType(Handle<crate::Expression>),
    #[error("Swizzle component {0:?} is outside of vector size {1:?}")]
    InvalidSwizzleComponent(crate::SwizzleComponent, crate::VectorSize),
    #[error(transparent)]
    Compose(#[from] ComposeError),
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
    #[error("Calling an undeclared function {0:?}")]
    CallToUndeclaredFunction(Handle<crate::Function>),
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
    #[error("Image other index parameter is misplaced")]
    InvalidImageOtherIndex,
    #[error("Image array index type of {0:?} is not an integer scalar")]
    InvalidImageArrayIndexType(Handle<crate::Expression>),
    #[error("Image other index type of {0:?} is not an integer scalar")]
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
    InvalidSampleOffset(crate::ImageDimension, Handle<crate::Constant>),
    #[error("Depth reference {0:?} is not a scalar float")]
    InvalidDepthReference(Handle<crate::Expression>),
    #[error("Sample level is not compatible with the image dimension {0:?}")]
    InvalidSampleLevel(crate::ImageDimension),
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
}

struct ExpressionTypeResolver<'a> {
    root: Handle<crate::Expression>,
    types: &'a Arena<crate::Type>,
    info: &'a FunctionInfo,
}

impl<'a> ExpressionTypeResolver<'a> {
    fn resolve(
        &self,
        handle: Handle<crate::Expression>,
    ) -> Result<&'a crate::TypeInner, ExpressionError> {
        if handle < self.root {
            Ok(self.info[handle].ty.inner_with(self.types))
        } else {
            Err(ExpressionError::ForwardDependency(handle))
        }
    }
}

impl super::Validator {
    pub(super) fn validate_expression(
        &self,
        root: Handle<crate::Expression>,
        expression: &crate::Expression,
        function: &crate::Function,
        module: &crate::Module,
        info: &FunctionInfo,
        other_infos: &[FunctionInfo],
    ) -> Result<ShaderStages, ExpressionError> {
        use crate::{Expression as E, ScalarKind as Sk, TypeInner as Ti};

        let resolver = ExpressionTypeResolver {
            root,
            types: &module.types,
            info,
        };

        let stages = match *expression {
            E::Access { base, index } => {
                match *resolver.resolve(base)? {
                    Ti::Vector { .. }
                    | Ti::Matrix { .. }
                    | Ti::Array { .. }
                    | Ti::Pointer { .. }
                    | Ti::ValuePointer { size: Some(_), .. } => {}
                    ref other => {
                        log::error!("Indexing of {:?}", other);
                        return Err(ExpressionError::InvalidBaseType(base));
                    }
                }
                match *resolver.resolve(index)? {
                    //TODO: only allow one of these
                    Ti::Scalar {
                        kind: Sk::Sint,
                        width: _,
                    }
                    | Ti::Scalar {
                        kind: Sk::Uint,
                        width: _,
                    } => {}
                    ref other => {
                        log::error!("Indexing by {:?}", other);
                        return Err(ExpressionError::InvalidIndexType(index));
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
                            size: crate::ArraySize::Constant(handle),
                            ..
                        } => module.constants[handle].to_array_length().unwrap(),
                        Ti::Array { .. } => !0, // can't statically know, but need run-time checks
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

                let limit = resolve_index_limit(module, base, resolver.resolve(base)?, true)?;
                if index >= limit {
                    return Err(ExpressionError::IndexOutOfBounds(base, index));
                }
                ShaderStages::all()
            }
            E::Constant(handle) => {
                let _ = module
                    .constants
                    .try_get(handle)
                    .ok_or(ExpressionError::ConstantDoesntExist(handle))?;
                ShaderStages::all()
            }
            E::Splat { size: _, value } => match *resolver.resolve(value)? {
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
                let vec_size = match *resolver.resolve(vector)? {
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
            E::Compose { ref components, ty } => {
                for &handle in components {
                    if handle >= root {
                        return Err(ExpressionError::ForwardDependency(handle));
                    }
                }
                validate_compose(
                    ty,
                    &module.constants,
                    &module.types,
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
            E::GlobalVariable(handle) => {
                let _ = module
                    .global_variables
                    .try_get(handle)
                    .ok_or(ExpressionError::GlobalVarDoesntExist(handle))?;
                ShaderStages::all()
            }
            E::LocalVariable(handle) => {
                let _ = function
                    .local_variables
                    .try_get(handle)
                    .ok_or(ExpressionError::LocalVarDoesntExist(handle))?;
                ShaderStages::all()
            }
            E::Load { pointer } => {
                match *resolver.resolve(pointer)? {
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
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                // check the validity of expressions
                let image_var = match function.expressions[image] {
                    crate::Expression::GlobalVariable(var_handle) => {
                        &module.global_variables[var_handle]
                    }
                    _ => return Err(ExpressionError::ExpectedGlobalVariable),
                };
                let sampler_var = match function.expressions[sampler] {
                    crate::Expression::GlobalVariable(var_handle) => {
                        &module.global_variables[var_handle]
                    }
                    _ => return Err(ExpressionError::ExpectedGlobalVariable),
                };
                let comparison = match module.types[sampler_var.ty].inner {
                    Ti::Sampler { comparison } => comparison,
                    _ => return Err(ExpressionError::ExpectedSamplerType(sampler_var.ty)),
                };

                let (class, dim) = match module.types[image_var.ty].inner {
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
                            match *resolver.resolve(expr)? {
                                Ti::Scalar {
                                    kind: Sk::Sint,
                                    width: _,
                                } => {}
                                _ => return Err(ExpressionError::InvalidImageArrayIndexType(expr)),
                            }
                        }
                        (class, dim)
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(image_var.ty)),
                };

                // check sampling and comparison properties
                let image_depth = match class {
                    crate::ImageClass::Sampled {
                        kind: crate::ScalarKind::Float,
                        multi: false,
                    } => false,
                    crate::ImageClass::Depth => true,
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
                match *resolver.resolve(coordinate)? {
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
                if let Some(const_handle) = offset {
                    let good = match module.constants[const_handle].inner {
                        crate::ConstantInner::Scalar {
                            width: _,
                            value: crate::ScalarValue::Sint(_),
                        } => num_components == 1,
                        crate::ConstantInner::Scalar { .. } => false,
                        crate::ConstantInner::Composite { ty, .. } => {
                            match module.types[ty].inner {
                                Ti::Vector {
                                    size,
                                    kind: Sk::Sint,
                                    ..
                                } => size as u32 == num_components,
                                _ => false,
                            }
                        }
                    };
                    if !good {
                        return Err(ExpressionError::InvalidSampleOffset(dim, const_handle));
                    }
                }

                // check depth reference type
                if let Some(expr) = depth_ref {
                    match *resolver.resolve(expr)? {
                        Ti::Scalar {
                            kind: Sk::Float, ..
                        } => {}
                        _ => return Err(ExpressionError::InvalidDepthReference(expr)),
                    }
                }

                // check level properties
                let can_level = match class {
                    crate::ImageClass::Sampled { multi, .. } => !multi,
                    crate::ImageClass::Storage { .. } => unreachable!(),
                    crate::ImageClass::Depth { .. } => true,
                };
                match level {
                    // require `can_level` here?
                    crate::SampleLevel::Auto => ShaderStages::FRAGMENT,
                    crate::SampleLevel::Zero => ShaderStages::all(),
                    crate::SampleLevel::Exact(expr) if can_level => {
                        match *resolver.resolve(expr)? {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidSampleLevelExactType(expr)),
                        }
                        ShaderStages::all()
                    }
                    crate::SampleLevel::Bias(expr) => {
                        match *resolver.resolve(expr)? {
                            Ti::Scalar {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidSampleLevelBiasType(expr)),
                        }
                        ShaderStages::all()
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        match *resolver.resolve(x)? {
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
                        match *resolver.resolve(y)? {
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
                    _ => return Err(ExpressionError::InvalidSampleLevel(dim)),
                }
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                let var = match function.expressions[image] {
                    crate::Expression::GlobalVariable(var_handle) => {
                        &module.global_variables[var_handle]
                    }
                    _ => return Err(ExpressionError::ExpectedGlobalVariable),
                };
                match module.types[var.ty].inner {
                    Ti::Image {
                        class,
                        arrayed,
                        dim,
                    } => {
                        match resolver.resolve(coordinate)?.image_storage_coordinates() {
                            Some(coord_dim) if coord_dim == dim => {}
                            _ => {
                                return Err(ExpressionError::InvalidImageCoordinateType(
                                    dim, coordinate,
                                ))
                            }
                        };
                        let needs_index = match class {
                            crate::ImageClass::Storage { .. } => false,
                            _ => true,
                        };
                        if arrayed != array_index.is_some() {
                            return Err(ExpressionError::InvalidImageArrayIndex);
                        }
                        if needs_index != index.is_some() {
                            return Err(ExpressionError::InvalidImageOtherIndex);
                        }
                        if let Some(expr) = array_index {
                            match *resolver.resolve(expr)? {
                                Ti::Scalar {
                                    kind: Sk::Sint,
                                    width: _,
                                } => {}
                                _ => return Err(ExpressionError::InvalidImageArrayIndexType(expr)),
                            }
                        }
                        if let Some(expr) = index {
                            match *resolver.resolve(expr)? {
                                Ti::Scalar {
                                    kind: Sk::Sint,
                                    width: _,
                                } => {}
                                _ => return Err(ExpressionError::InvalidImageOtherIndexType(expr)),
                            }
                        }
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(var.ty)),
                }
                ShaderStages::all()
            }
            E::ImageQuery { image, query } => {
                let var = match function.expressions[image] {
                    crate::Expression::GlobalVariable(var_handle) => {
                        &module.global_variables[var_handle]
                    }
                    _ => return Err(ExpressionError::ExpectedGlobalVariable),
                };
                match module.types[var.ty].inner {
                    Ti::Image { class, arrayed, .. } => {
                        let can_level = match class {
                            crate::ImageClass::Sampled { multi, .. } => !multi,
                            crate::ImageClass::Storage { .. } => false,
                            crate::ImageClass::Depth { .. } => true,
                        };
                        let good = match query {
                            crate::ImageQuery::NumLayers => arrayed,
                            crate::ImageQuery::Size { level: None } => true,
                            crate::ImageQuery::Size { level: Some(_) }
                            | crate::ImageQuery::NumLevels => can_level,
                            crate::ImageQuery::NumSamples => !can_level,
                        };
                        if !good {
                            return Err(ExpressionError::InvalidImageClass(class));
                        }
                    }
                    _ => return Err(ExpressionError::ExpectedImageType(var.ty)),
                }
                ShaderStages::all()
            }
            E::Unary { op, expr } => {
                use crate::UnaryOperator as Uo;
                let inner = resolver.resolve(expr)?;
                match (op, inner.scalar_kind()) {
                    (_, Some(Sk::Sint))
                    | (_, Some(Sk::Bool))
                    //TODO: restrict Negate for bools?
                    | (Uo::Negate, Some(Sk::Float))
                    | (Uo::Not, Some(Sk::Uint)) => {}
                    other => {
                        log::error!("Op {:?} kind {:?}", op, other);
                        return Err(ExpressionError::InvalidUnaryOperandType(op, expr));
                    }
                }
                ShaderStages::all()
            }
            E::Binary { op, left, right } => {
                use crate::BinaryOperator as Bo;
                let left_inner = resolver.resolve(left)?;
                let right_inner = resolver.resolve(right)?;
                let good = match op {
                    Bo::Add | Bo::Subtract | Bo::Divide | Bo::Modulo => match *left_inner {
                        Ti::Scalar { kind, .. } | Ti::Vector { kind, .. } => match kind {
                            Sk::Uint | Sk::Sint | Sk::Float => left_inner == right_inner,
                            Sk::Bool => false,
                        },
                        _ => false,
                    },
                    Bo::Multiply => {
                        let kind_match = match left_inner.scalar_kind() {
                            Some(Sk::Uint) | Some(Sk::Sint) | Some(Sk::Float) => true,
                            Some(Sk::Bool) | None => false,
                        };
                        //TODO: should we be more restrictive here? I.e. expect scalar only to the left.
                        let types_match = match (left_inner, right_inner) {
                            (&Ti::Scalar { kind: kind1, .. }, &Ti::Scalar { kind: kind2, .. })
                            | (&Ti::Vector { kind: kind1, .. }, &Ti::Scalar { kind: kind2, .. })
                            | (&Ti::Scalar { kind: kind1, .. }, &Ti::Vector { kind: kind2, .. }) => {
                                kind1 == kind2
                            }
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
                            (
                                &Ti::Matrix { columns, .. },
                                &Ti::Vector {
                                    kind: Sk::Float,
                                    size,
                                    ..
                                },
                            ) => columns == size,
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
                        kind_match && types_match && left_width == right_width
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
                    Bo::And | Bo::ExclusiveOr | Bo::InclusiveOr => match *left_inner {
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
                let accept_inner = resolver.resolve(accept)?;
                let reject_inner = resolver.resolve(reject)?;
                let condition_good = match *resolver.resolve(condition)? {
                    Ti::Scalar {
                        kind: Sk::Bool,
                        width: _,
                    } => accept_inner.is_sized(),
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
            E::Derivative { axis: _, expr } => {
                match *resolver.resolve(expr)? {
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
                let argument_inner = resolver.resolve(argument)?;
                match fun {
                    Rf::All | Rf::Any => match *argument_inner {
                        Ti::Vector { kind: Sk::Bool, .. } => {}
                        ref other => {
                            log::error!("All/Any of type {:?}", other);
                            return Err(ExpressionError::InvalidBooleanVector(argument));
                        }
                    },
                    Rf::IsNan | Rf::IsInf | Rf::IsFinite | Rf::IsNormal => match *argument_inner {
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
            } => {
                use crate::MathFunction as Mf;

                let arg_ty = resolver.resolve(arg)?;
                let arg1_ty = arg1.map(|expr| resolver.resolve(expr)).transpose()?;
                let arg2_ty = arg2.map(|expr| resolver.resolve(expr)).transpose()?;
                match fun {
                    Mf::Abs => {
                        if arg1_ty.is_some() | arg2_ty.is_some() {
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
                        let arg1_ty = match (arg1_ty, arg2_ty) {
                            (Some(ty1), None) => ty1,
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
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty) {
                            (Some(ty1), Some(ty2)) => (ty1, ty2),
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
                    Mf::Cos
                    | Mf::Cosh
                    | Mf::Sin
                    | Mf::Sinh
                    | Mf::Tan
                    | Mf::Tanh
                    | Mf::Acos
                    | Mf::Asin
                    | Mf::Atan
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
                    | Mf::Sign
                    | Mf::Sqrt
                    | Mf::InverseSqrt => {
                        if arg1_ty.is_some() | arg2_ty.is_some() {
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
                    Mf::Atan2 | Mf::Pow | Mf::Distance | Mf::Step => {
                        let arg1_ty = match (arg1_ty, arg2_ty) {
                            (Some(ty1), None) => ty1,
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
                    Mf::Modf | Mf::Frexp | Mf::Ldexp => {
                        let arg1_ty = match (arg1_ty, arg2_ty) {
                            (Some(ty1), None) => ty1,
                            _ => return Err(ExpressionError::WrongArgumentCount(fun)),
                        };
                        let (size0, width0) = match *arg_ty {
                            Ti::Scalar {
                                kind: Sk::Float,
                                width,
                            } => (None, width),
                            Ti::Vector {
                                kind: Sk::Float,
                                size,
                                width,
                            } => (Some(size), width),
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        };
                        let good = match *arg1_ty {
                            Ti::Pointer { base, class: _ } => module.types[base].inner == *arg_ty,
                            Ti::ValuePointer {
                                size,
                                kind: Sk::Float,
                                width,
                                class: _,
                            } => size == size0 && width == width0,
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
                    Mf::Dot | Mf::Outer | Mf::Cross | Mf::Reflect => {
                        let arg1_ty = match (arg1_ty, arg2_ty) {
                            (Some(ty1), None) => ty1,
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
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty) {
                            (Some(ty1), Some(ty2)) => (ty1, ty2),
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
                        if arg1_ty.is_some() | arg2_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Vector {
                                kind: Sk::Float, ..
                            } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::FaceForward | Mf::Fma | Mf::Mix | Mf::SmoothStep => {
                        let (arg1_ty, arg2_ty) = match (arg1_ty, arg2_ty) {
                            (Some(ty1), Some(ty2)) => (ty1, ty2),
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
                    Mf::Inverse | Mf::Determinant => {
                        if arg1_ty.is_some() | arg2_ty.is_some() {
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
                        if arg1_ty.is_some() | arg2_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Matrix { .. } => {}
                            _ => return Err(ExpressionError::InvalidArgumentType(fun, 0, arg)),
                        }
                    }
                    Mf::CountOneBits | Mf::ReverseBits => {
                        if arg1_ty.is_some() | arg2_ty.is_some() {
                            return Err(ExpressionError::WrongArgumentCount(fun));
                        }
                        match *arg_ty {
                            Ti::Scalar { kind: Sk::Sint, .. }
                            | Ti::Scalar { kind: Sk::Uint, .. }
                            | Ti::Vector { kind: Sk::Sint, .. }
                            | Ti::Vector { kind: Sk::Uint, .. } => {}
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
                let prev_kind = resolver
                    .resolve(expr)?
                    .scalar_kind()
                    .ok_or(ExpressionError::InvalidCastArgument)?;
                match convert {
                    Some(width) if !self.check_width(kind, width) => {
                        return Err(ExpressionError::InvalidCastArgument)
                    }
                    None if prev_kind == Sk::Bool || kind == Sk::Bool => {
                        return Err(ExpressionError::InvalidCastArgument)
                    }
                    _ => {}
                }
                ShaderStages::all()
            }
            E::Call(function) => other_infos[function.index()].available_stages,
            E::ArrayLength(expr) => match *resolver.resolve(expr)? {
                Ti::Pointer { base, .. } => {
                    if let Some(&Ti::Array {
                        size: crate::ArraySize::Dynamic,
                        ..
                    }) = resolver.types.try_get(base).map(|ty| &ty.inner)
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
        };
        Ok(stages)
    }
}
