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
            Self::Type((handle, _inner)) => ctx.format_type(handle),
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
        first_component_ty_inner: &'a crate::TypeInner,
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
        let constructor_h = self.constructor(constructor, ctx)?;

        let components = match *components {
            [] => Components::None,
            [component] => {
                let span = ctx.ast_expressions.get_span(component);
                let component = self.expression(component, ctx)?;
                let ty_inner = super::resolve_inner!(ctx, component);

                Components::One {
                    component,
                    span,
                    ty_inner,
                }
            }
            [component, ref rest @ ..] => {
                let span = ctx.ast_expressions.get_span(component);
                let component = self.expression(component, ctx)?;

                let components = std::iter::once(Ok(component))
                    .chain(
                        rest.iter()
                            .map(|&component| self.expression(component, ctx)),
                    )
                    .collect::<Result<_, _>>()?;
                let spans = std::iter::once(span)
                    .chain(
                        rest.iter()
                            .map(|&component| ctx.ast_expressions.get_span(component)),
                    )
                    .collect();

                let first_component_ty_inner = super::resolve_inner!(ctx, component);

                Components::Many {
                    components,
                    spans,
                    first_component_ty_inner,
                }
            }
        };

        // Even though we computed `constructor` above, wait until now to borrow
        // a reference to the `TypeInner`, so that the component-handling code
        // above can have mutable access to the type arena.
        let constructor = constructor_h.borrow_inner(ctx.module);

        let expr = match (components, constructor) {
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
                    return Err(Error::TypeNotInferrable(ty_span));
                }
            },

            // Scalar constructor & conversion (scalar -> scalar)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                Constructor::Type((_, &crate::TypeInner::Scalar { kind, width })),
            ) => crate::Expression::As {
                expr: component,
                kind,
                convert: Some(width),
            },

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
                        kind: dst_kind,
                        width: dst_width,
                    },
                )),
            ) if dst_size == src_size => crate::Expression::As {
                expr: component,
                kind: dst_kind,
                convert: Some(dst_width),
            },

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
                        width: dst_width,
                    },
                )),
            ) if dst_columns == src_columns && dst_rows == src_rows => crate::Expression::As {
                expr: component,
                kind: crate::ScalarKind::Float,
                convert: Some(dst_width),
            },

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
            ) => crate::Expression::Splat {
                size,
                value: component,
            },

            // Vector constructor (splat)
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Scalar {
                            kind: src_kind,
                            width: src_width,
                            ..
                        },
                    ..
                },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Vector {
                        size,
                        kind: dst_kind,
                        width: dst_width,
                    },
                )),
            ) if dst_kind == src_kind || dst_width == src_width => crate::Expression::Splat {
                size,
                value: component,
            },

            // Vector constructor (by elements)
            (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        &crate::TypeInner::Scalar { kind, width }
                        | &crate::TypeInner::Vector { kind, width, .. },
                    ..
                },
                Constructor::PartialVector { size },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        &crate::TypeInner::Scalar { .. } | &crate::TypeInner::Vector { .. },
                    ..
                },
                Constructor::Type((_, &crate::TypeInner::Vector { size, width, kind })),
            ) => {
                let inner = crate::TypeInner::Vector { size, kind, width };
                let ty = ctx.ensure_type_exists(inner);
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by elements)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Scalar { width, .. },
                    ..
                },
                Constructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                )),
            ) => {
                let vec_ty = ctx.ensure_type_exists(crate::TypeInner::Vector {
                    width,
                    kind: crate::ScalarKind::Float,
                    size: rows,
                });

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
                    width,
                });
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by columns)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Vector { width, .. },
                    ..
                },
                Constructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Vector { .. },
                    ..
                },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                )),
            ) => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
                crate::Expression::Compose { ty, components }
            }

            // Array constructor - infer type
            (components, Constructor::PartialArray) => {
                let components = components.into_components_vec();

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

                crate::Expression::Compose { ty, components }
            }

            // Array or Struct constructor
            (
                components,
                Constructor::Type((
                    ty,
                    &crate::TypeInner::Array { .. } | &crate::TypeInner::Struct { .. },
                )),
            ) => {
                let components = components.into_components_vec();
                crate::Expression::Compose { ty, components }
            }

            // ERRORS

            // Bad conversion (type cast)
            (Components::One { span, ty_inner, .. }, constructor) => {
                let from_type = ctx.format_typeinner(ty_inner);
                return Err(Error::BadTypeCast {
                    span,
                    from_type,
                    to_type: constructor.to_error_string(ctx),
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

            // Parameters are of the wrong type for vector or matrix constructor
            (
                Components::Many { spans, .. },
                Constructor::Type((
                    _,
                    &crate::TypeInner::Vector { .. } | &crate::TypeInner::Matrix { .. },
                ))
                | Constructor::PartialVector { .. }
                | Constructor::PartialMatrix { .. },
            ) => {
                return Err(Error::InvalidConstructorComponentType(spans[0], 0));
            }

            // Other types can't be constructed
            _ => return Err(Error::TypeNotConstructible(ty_span)),
        };

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
            ast::ConstructorType::Scalar { width, kind } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Scalar { width, kind });
                Constructor::Type(ty)
            }
            ast::ConstructorType::PartialVector { size } => Constructor::PartialVector { size },
            ast::ConstructorType::Vector { size, kind, width } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Vector { size, kind, width });
                Constructor::Type(ty)
            }
            ast::ConstructorType::PartialMatrix { columns, rows } => {
                Constructor::PartialMatrix { columns, rows }
            }
            ast::ConstructorType::Matrix {
                rows,
                columns,
                width,
            } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
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
