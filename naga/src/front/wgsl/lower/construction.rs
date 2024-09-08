use std::num::NonZeroU32;

use crate::front::wgsl::parse::ast;
use crate::{Handle, Span};

use crate::front::wgsl::error::Error;
use crate::front::wgsl::lower::{ExpressionContext, Lowerer};

/// A cooked form of `ast::ConstructorType` that uses Naga types whenever
/// possible.
enum Constructor<T> {
    /// A vector construction whose component type is inferred from the
    /// argument: `vec3(1.0)`.
    PartialVector { size: crate::VectorSize },

    /// A matrix construction whose component type is inferred from the
    /// argument: `mat2x2(1,2,3,4)`.
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },

    /// An array whose component type and size are inferred from the arguments:
    /// `array(3,4,5)`.
    PartialArray,

    /// A known Naga type.
    ///
    /// When we match on this type, we need to see the `TypeInner` here, but at
    /// the point that we build this value we'll still need mutable access to
    /// the module later. To avoid borrowing from the module, the type parameter
    /// `T` is `Handle<Type>` initially. Then we use `borrow_inner` to produce a
    /// version holding a tuple `(Handle<Type>, &TypeInner)`.
    Type(T),
}

impl Constructor<Handle<crate::Type>> {
    /// Return an equivalent `Constructor` value that includes borrowed
    /// `TypeInner` values alongside any type handles.
    ///
    /// The returned form is more convenient to match on, since the patterns
    /// can actually see what the handle refers to.
    fn borrow_inner(
        self,
        module: &crate::Module,
    ) -> Constructor<(Handle<crate::Type>, &crate::TypeInner)> {
        match self {
            Constructor::PartialVector { size } => Constructor::PartialVector { size },
            Constructor::PartialMatrix { columns, rows } => {
                Constructor::PartialMatrix { columns, rows }
            }
            Constructor::PartialArray => Constructor::PartialArray,
            Constructor::Type(handle) => Constructor::Type((handle, &module.types[handle].inner)),
        }
    }
}

impl Constructor<(Handle<crate::Type>, &crate::TypeInner)> {
    fn to_error_string(&self, ctx: &ExpressionContext) -> String {
        match *self {
            Self::PartialVector { size } => {
                format!("vec{}<?>", size as u32,)
            }
            Self::PartialMatrix { columns, rows } => {
                format!("mat{}x{}<?>", columns as u32, rows as u32,)
            }
            Self::PartialArray => "array<?, ?>".to_string(),
            Self::Type((handle, _inner)) => handle.to_wgsl(&ctx.module.to_ctx()),
        }
    }
}

enum Components<'a> {
    None,
    One {
        component: Handle<crate::Expression>,
        span: Span,
        ty_inner: &'a crate::TypeInner,
    },
    Many {
        components: Vec<Handle<crate::Expression>>,
        spans: Vec<Span>,
    },
}

impl Components<'_> {
    fn into_components_vec(self) -> Vec<Handle<crate::Expression>> {
        match self {
            Self::None => vec![],
            Self::One { component, .. } => vec![component],
            Self::Many { components, .. } => components,
        }
    }
}

impl<'source, 'temp> Lowerer<'source, 'temp> {
    /// Generate Naga IR for a type constructor expression.
    ///
    /// The `constructor` value represents the head of the constructor
    /// expression, which is at least a hint of which type is being built; if
    /// it's one of the `Partial` variants, we need to consider the argument
    /// types as well.
    ///
    /// This is used for [`Construct`] expressions, but also for [`Call`]
    /// expressions, once we've determined that the "callable" (in WGSL spec
    /// terms) is actually a type.
    ///
    /// [`Construct`]: ast::Expression::Construct
    /// [`Call`]: ast::Expression::Call
    pub fn construct(
        &mut self,
        span: Span,
        constructor: &ast::ConstructorType<'source>,
        ty_span: Span,
        components: &[Handle<ast::Expression<'source>>],
        ctx: &mut ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        use crate::proc::TypeResolution as Tr;

        let constructor_h = self.constructor(constructor, ctx)?;

        let components = match *components {
            [] => Components::None,
            [component] => {
                let span = ctx.ast_expressions.get_span(component);
                let component = self.expression_for_abstract(component, ctx)?;
                let ty_inner = super::resolve_inner!(ctx, component);

                Components::One {
                    component,
                    span,
                    ty_inner,
                }
            }
            ref ast_components @ [_, _, ..] => {
                let components = ast_components
                    .iter()
                    .map(|&expr| self.expression_for_abstract(expr, ctx))
                    .collect::<Result<_, _>>()?;
                let spans = ast_components
                    .iter()
                    .map(|&expr| ctx.ast_expressions.get_span(expr))
                    .collect();

                for &component in &components {
                    ctx.grow_types(component)?;
                }

                Components::Many { components, spans }
            }
        };

        // Even though we computed `constructor` above, wait until now to borrow
        // a reference to the `TypeInner`, so that the component-handling code
        // above can have mutable access to the type arena.
        let constructor = constructor_h.borrow_inner(ctx.module);

        let expr;
        match (components, constructor) {
            // Empty constructor
            (Components::None, dst_ty) => match dst_ty {
                Constructor::Type((result_ty, _)) => {
                    return ctx.append_expression(crate::Expression::ZeroValue(result_ty), span)
                }
                Constructor::PartialVector { .. }
                | Constructor::PartialMatrix { .. }
                | Constructor::PartialArray => {
                    // We have no arguments from which to infer the result type, so
                    // partial constructors aren't acceptable here.
                    return Err(Error::TypeNotInferable(ty_span));
                }
            },

            // Scalar constructor & conversion (scalar -> scalar)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                Constructor::Type((_, &crate::TypeInner::Scalar(scalar))),
            ) => {
                expr = crate::Expression::As {
                    expr: component,
                    kind: scalar.kind,
                    convert: Some(scalar.width),
                };
            }

            // Vector conversion (vector -> vector)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Vector { size: src_size, .. },
                    ..
                },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Vector {
                        size: dst_size,
                        scalar: dst_scalar,
                    },
                )),
            ) if dst_size == src_size => {
                expr = crate::Expression::As {
                    expr: component,
                    kind: dst_scalar.kind,
                    convert: Some(dst_scalar.width),
                };
            }

            // Vector conversion (vector -> vector) - partial
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Vector { size: src_size, .. },
                    ..
                },
                Constructor::PartialVector { size: dst_size },
            ) if dst_size == src_size => {
                // This is a trivial conversion: the sizes match, and a Partial
                // constructor doesn't specify a scalar type, so nothing can
                // possibly happen.
                return Ok(component);
            }

            // Matrix conversion (matrix -> matrix)
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Matrix {
                            columns: src_columns,
                            rows: src_rows,
                            ..
                        },
                    ..
                },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Matrix {
                        columns: dst_columns,
                        rows: dst_rows,
                        scalar: dst_scalar,
                    },
                )),
            ) if dst_columns == src_columns && dst_rows == src_rows => {
                expr = crate::Expression::As {
                    expr: component,
                    kind: dst_scalar.kind,
                    convert: Some(dst_scalar.width),
                };
            }

            // Matrix conversion (matrix -> matrix) - partial
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Matrix {
                            columns: src_columns,
                            rows: src_rows,
                            ..
                        },
                    ..
                },
                Constructor::PartialMatrix {
                    columns: dst_columns,
                    rows: dst_rows,
                },
            ) if dst_columns == src_columns && dst_rows == src_rows => {
                // This is a trivial conversion: the sizes match, and a Partial
                // constructor doesn't specify a scalar type, so nothing can
                // possibly happen.
                return Ok(component);
            }

            // Vector constructor (splat) - infer type
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                Constructor::PartialVector { size },
            ) => {
                expr = crate::Expression::Splat {
                    size,
                    value: component,
                };
            }

            // Vector constructor (splat)
            (
                Components::One {
                    mut component,
                    ty_inner: &crate::TypeInner::Scalar(_),
                    ..
                },
                Constructor::Type((_, &crate::TypeInner::Vector { size, scalar })),
            ) => {
                ctx.convert_slice_to_common_leaf_scalar(
                    std::slice::from_mut(&mut component),
                    scalar,
                )?;
                expr = crate::Expression::Splat {
                    size,
                    value: component,
                };
            }

            // Vector constructor (by elements), partial
            (
                Components::Many {
                    mut components,
                    spans,
                },
                Constructor::PartialVector { size },
            ) => {
                let consensus_scalar =
                    ctx.automatic_conversion_consensus(&components)
                        .map_err(|index| {
                            Error::InvalidConstructorComponentType(spans[index], index as i32)
                        })?;
                ctx.convert_slice_to_common_leaf_scalar(&mut components, consensus_scalar)?;
                let inner = consensus_scalar.to_inner_vector(size);
                let ty = ctx.ensure_type_exists(inner);
                expr = crate::Expression::Compose { ty, components };
            }

            // Vector constructor (by elements), full type given
            (
                Components::Many { mut components, .. },
                Constructor::Type((ty, &crate::TypeInner::Vector { scalar, .. })),
            ) => {
                ctx.try_automatic_conversions_for_vector(&mut components, scalar, ty_span)?;
                expr = crate::Expression::Compose { ty, components };
            }

            // Matrix constructor (by elements), partial
            (
                Components::Many {
                    mut components,
                    spans,
                },
                Constructor::PartialMatrix { columns, rows },
            ) if components.len() == columns as usize * rows as usize => {
                let consensus_scalar =
                    ctx.automatic_conversion_consensus(&components)
                        .map_err(|index| {
                            Error::InvalidConstructorComponentType(spans[index], index as i32)
                        })?;
                // We actually only accept floating-point elements.
                let consensus_scalar = consensus_scalar
                    .automatic_conversion_combine(crate::Scalar::ABSTRACT_FLOAT)
                    .unwrap_or(consensus_scalar);
                ctx.convert_slice_to_common_leaf_scalar(&mut components, consensus_scalar)?;
                let vec_ty = ctx.ensure_type_exists(consensus_scalar.to_inner_vector(rows));

                let components = components
                    .chunks(rows as usize)
                    .map(|vec_components| {
                        ctx.append_expression(
                            crate::Expression::Compose {
                                ty: vec_ty,
                                components: Vec::from(vec_components),
                            },
                            Default::default(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar: consensus_scalar,
                });
                expr = crate::Expression::Compose { ty, components };
            }

            // Matrix constructor (by elements), type given
            (
                Components::Many { mut components, .. },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    },
                )),
            ) if components.len() == columns as usize * rows as usize => {
                let element = Tr::Value(crate::TypeInner::Scalar(scalar));
                ctx.try_automatic_conversions_slice(&mut components, &element, ty_span)?;
                let vec_ty = ctx.ensure_type_exists(scalar.to_inner_vector(rows));

                let components = components
                    .chunks(rows as usize)
                    .map(|vec_components| {
                        ctx.append_expression(
                            crate::Expression::Compose {
                                ty: vec_ty,
                                components: Vec::from(vec_components),
                            },
                            Default::default(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                });
                expr = crate::Expression::Compose { ty, components };
            }

            // Matrix constructor (by columns), partial
            (
                Components::Many {
                    mut components,
                    spans,
                },
                Constructor::PartialMatrix { columns, rows },
            ) => {
                let consensus_scalar =
                    ctx.automatic_conversion_consensus(&components)
                        .map_err(|index| {
                            Error::InvalidConstructorComponentType(spans[index], index as i32)
                        })?;
                ctx.convert_slice_to_common_leaf_scalar(&mut components, consensus_scalar)?;
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar: consensus_scalar,
                });
                expr = crate::Expression::Compose { ty, components };
            }

            // Matrix constructor (by columns), type given
            (
                Components::Many { mut components, .. },
                Constructor::Type((
                    ty,
                    &crate::TypeInner::Matrix {
                        columns: _,
                        rows,
                        scalar,
                    },
                )),
            ) => {
                let component_ty = crate::TypeInner::Vector { size: rows, scalar };
                ctx.try_automatic_conversions_slice(
                    &mut components,
                    &Tr::Value(component_ty),
                    ty_span,
                )?;
                expr = crate::Expression::Compose { ty, components };
            }

            // Array constructor - infer type
            (components, Constructor::PartialArray) => {
                let mut components = components.into_components_vec();
                if let Ok(consensus_scalar) = ctx.automatic_conversion_consensus(&components) {
                    // Note that this will *not* necessarily convert all the
                    // components to the same type! The `automatic_conversion_consensus`
                    // method only considers the parameters' leaf scalar
                    // types; the parameters themselves could be any mix of
                    // vectors, matrices, and scalars.
                    //
                    // But *if* it is possible for this array construction
                    // expression to be well-typed at all, then all the
                    // parameters must have the same type constructors (vec,
                    // matrix, scalar) applied to their leaf scalars, so
                    // reconciling their scalars is always the right thing to
                    // do. And if this array construction is not well-typed,
                    // these conversions will not make it so, and we can let
                    // validation catch the error.
                    ctx.convert_slice_to_common_leaf_scalar(&mut components, consensus_scalar)?;
                } else {
                    // There's no consensus scalar. Emit the `Compose`
                    // expression anyway, and let validation catch the problem.
                }

                let base = ctx.register_type(components[0])?;

                let inner = crate::TypeInner::Array {
                    base,
                    size: crate::ArraySize::Constant(
                        NonZeroU32::new(u32::try_from(components.len()).unwrap()).unwrap(),
                    ),
                    stride: {
                        self.layouter.update(ctx.module.to_ctx()).unwrap();
                        self.layouter[base].to_stride()
                    },
                };
                let ty = ctx.ensure_type_exists(inner);

                expr = crate::Expression::Compose { ty, components };
            }

            // Array constructor, explicit type
            (components, Constructor::Type((ty, &crate::TypeInner::Array { base, .. }))) => {
                let mut components = components.into_components_vec();
                ctx.try_automatic_conversions_slice(&mut components, &Tr::Handle(base), ty_span)?;
                expr = crate::Expression::Compose { ty, components };
            }

            // Struct constructor
            (
                components,
                Constructor::Type((ty, &crate::TypeInner::Struct { ref members, .. })),
            ) => {
                let mut components = components.into_components_vec();
                let struct_ty_span = ctx.module.types.get_span(ty);

                // Make a vector of the members' type handles in advance, to
                // avoid borrowing `members` from `ctx` while we generate
                // new code.
                let members: Vec<Handle<crate::Type>> = members.iter().map(|m| m.ty).collect();

                for (component, &ty) in components.iter_mut().zip(&members) {
                    *component =
                        ctx.try_automatic_conversions(*component, &Tr::Handle(ty), struct_ty_span)?;
                }
                expr = crate::Expression::Compose { ty, components };
            }

            // ERRORS

            // Bad conversion (type cast)
            (Components::One { span, ty_inner, .. }, constructor) => {
                let from_type = ty_inner.to_wgsl(&ctx.module.to_ctx()).into();
                return Err(Error::BadTypeCast {
                    span,
                    from_type,
                    to_type: constructor.to_error_string(ctx).into(),
                });
            }

            // Too many parameters for scalar constructor
            (
                Components::Many { spans, .. },
                Constructor::Type((_, &crate::TypeInner::Scalar { .. })),
            ) => {
                let span = spans[1].until(spans.last().unwrap());
                return Err(Error::UnexpectedComponents(span));
            }

            // Other types can't be constructed
            _ => return Err(Error::TypeNotConstructible(ty_span)),
        }

        let expr = ctx.append_expression(expr, span)?;
        Ok(expr)
    }

    /// Build a [`Constructor`] for a WGSL construction expression.
    ///
    /// If `constructor` conveys enough information to determine which Naga [`Type`]
    /// we're actually building (i.e., it's not a partial constructor), then
    /// ensure the `Type` exists in [`ctx.module`], and return
    /// [`Constructor::Type`].
    ///
    /// Otherwise, return the [`Constructor`] partial variant corresponding to
    /// `constructor`.
    ///
    /// [`Type`]: crate::Type
    /// [`ctx.module`]: ExpressionContext::module
    fn constructor<'out>(
        &mut self,
        constructor: &ast::ConstructorType<'source>,
        ctx: &mut ExpressionContext<'source, '_, 'out>,
    ) -> Result<Constructor<Handle<crate::Type>>, Error<'source>> {
        let handle = match *constructor {
            ast::ConstructorType::Scalar(scalar) => {
                let ty = ctx.ensure_type_exists(scalar.to_inner_scalar());
                Constructor::Type(ty)
            }
            ast::ConstructorType::PartialVector { size } => Constructor::PartialVector { size },
            ast::ConstructorType::Vector { size, ty, ty_span } => {
                let ty = self.resolve_ast_type(ty, &mut ctx.as_global())?;
                let scalar = match ctx.module.types[ty].inner {
                    crate::TypeInner::Scalar(sc) => sc,
                    _ => return Err(Error::UnknownScalarType(ty_span)),
                };
                let ty = ctx.ensure_type_exists(crate::TypeInner::Vector { size, scalar });
                Constructor::Type(ty)
            }
            ast::ConstructorType::PartialMatrix { columns, rows } => {
                Constructor::PartialMatrix { columns, rows }
            }
            ast::ConstructorType::Matrix {
                rows,
                columns,
                ty,
                ty_span,
            } => {
                let ty = self.resolve_ast_type(ty, &mut ctx.as_global())?;
                let scalar = match ctx.module.types[ty].inner {
                    crate::TypeInner::Scalar(sc) => sc,
                    _ => return Err(Error::UnknownScalarType(ty_span)),
                };
                let ty = match scalar.kind {
                    crate::ScalarKind::Float => ctx.ensure_type_exists(crate::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    }),
                    _ => return Err(Error::BadMatrixScalarKind(ty_span, scalar)),
                };
                Constructor::Type(ty)
            }
            ast::ConstructorType::PartialArray => Constructor::PartialArray,
            ast::ConstructorType::Array { base, size } => {
                let base = self.resolve_ast_type(base, &mut ctx.as_global())?;
                let size = self.array_size(size, &mut ctx.as_global())?;

                self.layouter.update(ctx.module.to_ctx()).unwrap();
                let stride = self.layouter[base].to_stride();

                let ty = ctx.ensure_type_exists(crate::TypeInner::Array { base, size, stride });
                Constructor::Type(ty)
            }
            ast::ConstructorType::Type(ty) => Constructor::Type(ty),
        };

        Ok(handle)
    }
}
