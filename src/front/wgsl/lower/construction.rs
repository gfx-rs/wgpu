use std::num::NonZeroU32;

use crate::front::wgsl::parse::ast;
use crate::{Handle, Span};

use crate::front::wgsl::error::Error;
use crate::front::wgsl::lower::{ExpressionContext, Lowerer};
use crate::proc::TypeResolution;

enum ConcreteConstructorHandle {
    PartialVector {
        size: crate::VectorSize,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    PartialArray,
    Type(Handle<crate::Type>),
}

impl ConcreteConstructorHandle {
    fn borrow<'a>(&self, module: &'a crate::Module) -> ConcreteConstructor<'a> {
        match *self {
            Self::PartialVector { size } => ConcreteConstructor::PartialVector { size },
            Self::PartialMatrix { columns, rows } => {
                ConcreteConstructor::PartialMatrix { columns, rows }
            }
            Self::PartialArray => ConcreteConstructor::PartialArray,
            Self::Type(handle) => ConcreteConstructor::Type(handle, &module.types[handle].inner),
        }
    }
}

enum ConcreteConstructor<'a> {
    PartialVector {
        size: crate::VectorSize,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    PartialArray,
    Type(Handle<crate::Type>, &'a crate::TypeInner),
}

impl ConcreteConstructorHandle {
    fn to_error_string(&self, ctx: ExpressionContext) -> String {
        match *self {
            Self::PartialVector { size } => {
                format!("vec{}<?>", size as u32,)
            }
            Self::PartialMatrix { columns, rows } => {
                format!("mat{}x{}<?>", columns as u32, rows as u32,)
            }
            Self::PartialArray => "array<?, ?>".to_string(),
            Self::Type(ty) => ctx.format_type(ty),
        }
    }
}

enum ComponentsHandle<'a> {
    None,
    One {
        component: Handle<crate::Expression>,
        span: Span,
        ty: &'a TypeResolution,
    },
    Many {
        components: Vec<Handle<crate::Expression>>,
        spans: Vec<Span>,
        first_component_ty: &'a TypeResolution,
    },
}

impl<'a> ComponentsHandle<'a> {
    fn borrow(self, module: &'a crate::Module) -> Components<'a> {
        match self {
            Self::None => Components::None,
            Self::One {
                component,
                span,
                ty,
            } => Components::One {
                component,
                span,
                ty_inner: ty.inner_with(&module.types),
            },
            Self::Many {
                components,
                spans,
                first_component_ty,
            } => Components::Many {
                components,
                spans,
                first_component_ty_inner: first_component_ty.inner_with(&module.types),
            },
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
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let constructor_h = self.constructor(constructor, ctx.reborrow())?;

        let components_h = match *components {
            [] => ComponentsHandle::None,
            [component] => {
                let span = ctx.ast_expressions.get_span(component);
                let component = self.expression(component, ctx.reborrow())?;
                ctx.grow_types(component)?;
                let ty = &ctx.typifier()[component];

                ComponentsHandle::One {
                    component,
                    span,
                    ty,
                }
            }
            [component, ref rest @ ..] => {
                let span = ctx.ast_expressions.get_span(component);
                let component = self.expression(component, ctx.reborrow())?;

                let components = std::iter::once(Ok(component))
                    .chain(
                        rest.iter()
                            .map(|&component| self.expression(component, ctx.reborrow())),
                    )
                    .collect::<Result<_, _>>()?;
                let spans = std::iter::once(span)
                    .chain(
                        rest.iter()
                            .map(|&component| ctx.ast_expressions.get_span(component)),
                    )
                    .collect();

                ctx.grow_types(component)?;
                let ty = &ctx.typifier()[component];

                ComponentsHandle::Many {
                    components,
                    spans,
                    first_component_ty: ty,
                }
            }
        };

        let (components, constructor) = (
            components_h.borrow(ctx.module),
            constructor_h.borrow(ctx.module),
        );
        let expr = match (components, constructor) {
            // Empty constructor
            (Components::None, dst_ty) => match dst_ty {
                ConcreteConstructor::Type(ty, _) => {
                    return ctx.append_expression(crate::Expression::ZeroValue(ty), span)
                }
                _ => return Err(Error::TypeNotInferrable(ty_span)),
            },

            // Scalar constructor & conversion (scalar -> scalar)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(_, &crate::TypeInner::Scalar { kind, width }),
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
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector {
                        size: dst_size,
                        kind: dst_kind,
                        width: dst_width,
                    },
                ),
            ) if dst_size == src_size => crate::Expression::As {
                expr: component,
                kind: dst_kind,
                convert: Some(dst_width),
            },

            // Vector conversion (vector -> vector) - partial
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Vector {
                            size: src_size,
                            kind: src_kind,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::PartialVector { size: dst_size },
            ) if dst_size == src_size => crate::Expression::As {
                expr: component,
                kind: src_kind,
                convert: None,
            },

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
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns: dst_columns,
                        rows: dst_rows,
                        width: dst_width,
                    },
                ),
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
                ConcreteConstructor::PartialMatrix {
                    columns: dst_columns,
                    rows: dst_rows,
                },
            ) if dst_columns == src_columns && dst_rows == src_rows => crate::Expression::As {
                expr: component,
                kind: crate::ScalarKind::Float,
                convert: None,
            },

            // Vector constructor (splat) - infer type
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::PartialVector { size },
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
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector {
                        size,
                        kind: dst_kind,
                        width: dst_width,
                    },
                ),
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
                ConcreteConstructor::PartialVector { size },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        &crate::TypeInner::Scalar { .. } | &crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(_, &crate::TypeInner::Vector { size, width, kind }),
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
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
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
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
            ) => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
                crate::Expression::Compose { ty, components }
            }

            // Array constructor - infer type
            (components, ConcreteConstructor::PartialArray) => {
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

            // Array constructor
            (components, ConcreteConstructor::Type(ty, &crate::TypeInner::Array { .. })) => {
                let components = components.into_components_vec();
                crate::Expression::Compose { ty, components }
            }

            // Struct constructor
            (components, ConcreteConstructor::Type(ty, &crate::TypeInner::Struct { .. })) => {
                crate::Expression::Compose {
                    ty,
                    components: components.into_components_vec(),
                }
            }

            // ERRORS

            // Bad conversion (type cast)
            (Components::One { span, ty_inner, .. }, _) => {
                let from_type = ctx.format_typeinner(ty_inner);
                return Err(Error::BadTypeCast {
                    span,
                    from_type,
                    to_type: constructor_h.to_error_string(ctx.reborrow()),
                });
            }

            // Too many parameters for scalar constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(_, &crate::TypeInner::Scalar { .. }),
            ) => {
                let span = spans[1].until(spans.last().unwrap());
                return Err(Error::UnexpectedComponents(span));
            }

            // Parameters are of the wrong type for vector or matrix constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector { .. } | &crate::TypeInner::Matrix { .. },
                )
                | ConcreteConstructor::PartialVector { .. }
                | ConcreteConstructor::PartialMatrix { .. },
            ) => {
                return Err(Error::InvalidConstructorComponentType(spans[0], 0));
            }

            // Other types can't be constructed
            _ => return Err(Error::TypeNotConstructible(ty_span)),
        };

        let expr = ctx.append_expression(expr, span)?;
        Ok(expr)
    }

    /// Build a Naga IR [`Type`] for `constructor` if there is enough
    /// information to do so.
    ///
    /// For `Partial` variants of [`ast::ConstructorType`], we don't know the
    /// component type, so in that case we return the appropriate `Partial`
    /// variant of [`ConcreteConstructorHandle`].
    ///
    /// But for the other `ConstructorType` variants, we have everything we need
    /// to know to actually produce a Naga IR type. In this case we add to/find
    /// in [`ctx.module`] a suitable Naga `Type` and return a
    /// [`ConcreteConstructorHandle::Type`] value holding its handle.
    ///
    /// Note that constructing an [`Array`] type may require inserting
    /// [`Constant`]s as well as `Type`s into `ctx.module`, to represent the
    /// array's length.
    ///
    /// [`Type`]: crate::Type
    /// [`ctx.module`]: ExpressionContext::module
    /// [`Array`]: crate::TypeInner::Array
    /// [`Constant`]: crate::Constant
    fn constructor<'out>(
        &mut self,
        constructor: &ast::ConstructorType<'source>,
        mut ctx: ExpressionContext<'source, '_, 'out>,
    ) -> Result<ConcreteConstructorHandle, Error<'source>> {
        let c = match *constructor {
            ast::ConstructorType::Scalar { width, kind } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Scalar { width, kind });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialVector { size } => {
                ConcreteConstructorHandle::PartialVector { size }
            }
            ast::ConstructorType::Vector { size, kind, width } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Vector { size, kind, width });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialMatrix { rows, columns } => {
                ConcreteConstructorHandle::PartialMatrix { rows, columns }
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
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialArray => ConcreteConstructorHandle::PartialArray,
            ast::ConstructorType::Array { base, size } => {
                let base = self.resolve_ast_type(base, ctx.as_global())?;
                let size = match size {
                    ast::ArraySize::Constant(expr) => {
                        let const_expr = self.expression(expr, ctx.as_const())?;
                        crate::ArraySize::Constant(ctx.as_const().array_length(const_expr)?)
                    }
                    ast::ArraySize::Dynamic => crate::ArraySize::Dynamic,
                };

                self.layouter.update(ctx.module.to_ctx()).unwrap();
                let ty = ctx.ensure_type_exists(crate::TypeInner::Array {
                    base,
                    size,
                    stride: self.layouter[base].to_stride(),
                });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::Type(ty) => ConcreteConstructorHandle::Type(ty),
        };

        Ok(c)
    }
}
