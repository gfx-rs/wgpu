use crate::{
    proc::TypeResolution, Arena, ArraySize, Bytes, Constant, ConstantInner, Expression, Handle,
    ScalarKind, ScalarValue, Span as NagaSpan, Type, TypeInner, UniqueArena, VectorSize,
};

use super::{Error, ExpressionContext, Lexer, Parser, Scope, Span, Token};

/// Represents the type of the constructor
///
/// Vectors, Matrices and Arrays can have partial type information
/// which later gets inferred from the constructor parameters
enum ConstructorType {
    Scalar {
        kind: ScalarKind,
        width: Bytes,
    },
    PartialVector {
        size: VectorSize,
    },
    Vector {
        size: VectorSize,
        kind: ScalarKind,
        width: Bytes,
    },
    PartialMatrix {
        columns: VectorSize,
        rows: VectorSize,
    },
    Matrix {
        columns: VectorSize,
        rows: VectorSize,
        width: Bytes,
    },
    PartialArray,
    Array {
        base: Handle<Type>,
        size: ArraySize,
        stride: u32,
    },
    Struct(Handle<Type>),
}

impl ConstructorType {
    const fn to_type_resolution(&self) -> Option<TypeResolution> {
        Some(match *self {
            ConstructorType::Scalar { kind, width } => {
                TypeResolution::Value(TypeInner::Scalar { kind, width })
            }
            ConstructorType::Vector { size, kind, width } => {
                TypeResolution::Value(TypeInner::Vector { size, kind, width })
            }
            ConstructorType::Matrix {
                columns,
                rows,
                width,
            } => TypeResolution::Value(TypeInner::Matrix {
                columns,
                rows,
                width,
            }),
            ConstructorType::Array { base, size, stride } => {
                TypeResolution::Value(TypeInner::Array { base, size, stride })
            }
            ConstructorType::Struct(handle) => TypeResolution::Handle(handle),
            _ => return None,
        })
    }
}

impl ConstructorType {
    fn to_error_string(&self, types: &UniqueArena<Type>, constants: &Arena<Constant>) -> String {
        match *self {
            ConstructorType::Scalar { kind, width } => kind.to_wgsl(width),
            ConstructorType::PartialVector { size } => {
                format!("vec{}<?>", size as u32,)
            }
            ConstructorType::Vector { size, kind, width } => {
                format!("vec{}<{}>", size as u32, kind.to_wgsl(width))
            }
            ConstructorType::PartialMatrix { columns, rows } => {
                format!("mat{}x{}<?>", columns as u32, rows as u32,)
            }
            ConstructorType::Matrix {
                columns,
                rows,
                width,
            } => {
                format!(
                    "mat{}x{}<{}>",
                    columns as u32,
                    rows as u32,
                    ScalarKind::Float.to_wgsl(width)
                )
            }
            ConstructorType::PartialArray => "array<?, ?>".to_string(),
            ConstructorType::Array { base, size, .. } => {
                format!(
                    "array<{}, {}>",
                    types[base].name.as_deref().unwrap_or("?"),
                    match size {
                        ArraySize::Constant(size) => {
                            constants[size]
                                .to_array_length()
                                .map(|len| len.to_string())
                                .unwrap_or_else(|| "?".to_string())
                        }
                        _ => unreachable!(),
                    }
                )
            }
            ConstructorType::Struct(handle) => types[handle]
                .name
                .clone()
                .unwrap_or_else(|| "?".to_string()),
        }
    }
}

fn parse_constructor_type<'a>(
    parser: &mut Parser,
    lexer: &mut Lexer<'a>,
    word: &'a str,
    type_arena: &mut UniqueArena<Type>,
    const_arena: &mut Arena<Constant>,
) -> Result<Option<ConstructorType>, Error<'a>> {
    if let Some((kind, width)) = super::conv::get_scalar_type(word) {
        return Ok(Some(ConstructorType::Scalar { kind, width }));
    }

    let partial = match word {
        "vec2" => ConstructorType::PartialVector {
            size: VectorSize::Bi,
        },
        "vec3" => ConstructorType::PartialVector {
            size: VectorSize::Tri,
        },
        "vec4" => ConstructorType::PartialVector {
            size: VectorSize::Quad,
        },
        "mat2x2" => ConstructorType::PartialMatrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Bi,
        },
        "mat2x3" => ConstructorType::PartialMatrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Tri,
        },
        "mat2x4" => ConstructorType::PartialMatrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Quad,
        },
        "mat3x2" => ConstructorType::PartialMatrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Bi,
        },
        "mat3x3" => ConstructorType::PartialMatrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Tri,
        },
        "mat3x4" => ConstructorType::PartialMatrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Quad,
        },
        "mat4x2" => ConstructorType::PartialMatrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Bi,
        },
        "mat4x3" => ConstructorType::PartialMatrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
        },
        "mat4x4" => ConstructorType::PartialMatrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
        },
        "array" => ConstructorType::PartialArray,
        _ => return Ok(None),
    };

    // parse component type if present
    match (lexer.peek().0, partial) {
        (Token::Paren('<'), ConstructorType::PartialVector { size }) => {
            let (kind, width) = lexer.next_scalar_generic()?;
            Ok(Some(ConstructorType::Vector { size, kind, width }))
        }
        (Token::Paren('<'), ConstructorType::PartialMatrix { columns, rows }) => {
            let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
            match kind {
                ScalarKind::Float => Ok(Some(ConstructorType::Matrix {
                    columns,
                    rows,
                    width,
                })),
                _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
            }
        }
        (Token::Paren('<'), ConstructorType::PartialArray) => {
            lexer.expect_generic_paren('<')?;
            let base = parser.parse_type_decl(lexer, None, type_arena, const_arena)?;
            let size = if lexer.skip(Token::Separator(',')) {
                let const_handle = parser.parse_const_expression(lexer, type_arena, const_arena)?;
                ArraySize::Constant(const_handle)
            } else {
                ArraySize::Dynamic
            };
            lexer.expect_generic_paren('>')?;

            let stride = {
                parser.layouter.update(type_arena, const_arena).unwrap();
                parser.layouter[base].to_stride()
            };

            Ok(Some(ConstructorType::Array { base, size, stride }))
        }
        (_, partial) => Ok(Some(partial)),
    }
}

/// Expects [`Scope::PrimaryExpr`] scope on top; if returning Some(_), pops it.
pub(super) fn parse_construction<'a>(
    parser: &mut Parser,
    lexer: &mut Lexer<'a>,
    type_name: &'a str,
    type_span: Span,
    mut ctx: ExpressionContext<'a, '_, '_>,
) -> Result<Option<Handle<Expression>>, Error<'a>> {
    assert_eq!(
        parser.scopes.last().map(|&(ref scope, _)| scope.clone()),
        Some(Scope::PrimaryExpr)
    );
    let dst_ty = match parser.lookup_type.get(type_name) {
        Some(&handle) => ConstructorType::Struct(handle),
        None => match parse_constructor_type(parser, lexer, type_name, ctx.types, ctx.constants)? {
            Some(inner) => inner,
            None => {
                match parser.parse_type_decl_impl(
                    lexer,
                    super::TypeAttributes::default(),
                    type_name,
                    ctx.types,
                    ctx.constants,
                )? {
                    Some(_) => {
                        return Err(Error::TypeNotConstructible(type_span));
                    }
                    None => return Ok(None),
                }
            }
        },
    };

    lexer.open_arguments()?;

    let mut components = Vec::new();
    let mut spans = Vec::new();

    if lexer.peek().0 == Token::Paren(')') {
        let _ = lexer.next();
    } else {
        while components.is_empty() || lexer.next_argument()? {
            let (component, span) = lexer
                .capture_span(|lexer| parser.parse_general_expression(lexer, ctx.reborrow()))?;
            components.push(component);
            spans.push(span);
        }
    }

    enum Components<'a> {
        None,
        One {
            component: Handle<Expression>,
            span: Span,
            ty: &'a TypeInner,
        },
        Many {
            components: Vec<Handle<Expression>>,
            spans: Vec<Span>,
            first_component_ty: &'a TypeInner,
        },
    }

    impl<'a> Components<'a> {
        fn into_components_vec(self) -> Vec<Handle<Expression>> {
            match self {
                Components::None => vec![],
                Components::One { component, .. } => vec![component],
                Components::Many { components, .. } => components,
            }
        }
    }

    let components = match *components.as_slice() {
        [] => Components::None,
        [component] => {
            ctx.resolve_type(component)?;
            Components::One {
                component,
                span: spans[0].clone(),
                ty: ctx.typifier.get(component, ctx.types),
            }
        }
        [component, ..] => {
            ctx.resolve_type(component)?;
            Components::Many {
                components,
                spans,
                first_component_ty: ctx.typifier.get(component, ctx.types),
            }
        }
    };

    let expr = match (components, dst_ty) {
        // Empty constructor
        (Components::None, dst_ty) => {
            let ty = match dst_ty.to_type_resolution() {
                Some(TypeResolution::Handle(handle)) => handle,
                Some(TypeResolution::Value(inner)) => ctx
                    .types
                    .insert(Type { name: None, inner }, Default::default()),
                None => return Err(Error::TypeNotInferrable(type_span)),
            };

            return match ctx.create_zero_value_constant(ty) {
                Some(constant) => {
                    let span = parser.pop_scope(lexer);
                    Ok(Some(ctx.interrupt_emitter(
                        Expression::Constant(constant),
                        span.into(),
                    )))
                }
                None => Err(Error::TypeNotConstructible(type_span)),
            };
        }

        // Scalar constructor & conversion (scalar -> scalar)
        (
            Components::One {
                component,
                ty: &TypeInner::Scalar { .. },
                ..
            },
            ConstructorType::Scalar { kind, width },
        ) => Expression::As {
            expr: component,
            kind,
            convert: Some(width),
        },

        // Vector conversion (vector -> vector)
        (
            Components::One {
                component,
                ty: &TypeInner::Vector { size: src_size, .. },
                ..
            },
            ConstructorType::Vector {
                size: dst_size,
                kind: dst_kind,
                width: dst_width,
            },
        ) if dst_size == src_size => Expression::As {
            expr: component,
            kind: dst_kind,
            convert: Some(dst_width),
        },

        // Vector conversion (vector -> vector) - partial
        (
            Components::One {
                component,
                ty:
                    &TypeInner::Vector {
                        size: src_size,
                        kind: src_kind,
                        ..
                    },
                ..
            },
            ConstructorType::PartialVector { size: dst_size },
        ) if dst_size == src_size => Expression::As {
            expr: component,
            kind: src_kind,
            convert: None,
        },

        // Matrix conversion (matrix -> matrix)
        (
            Components::One {
                component,
                ty:
                    &TypeInner::Matrix {
                        columns: src_columns,
                        rows: src_rows,
                        ..
                    },
                ..
            },
            ConstructorType::Matrix {
                columns: dst_columns,
                rows: dst_rows,
                width: dst_width,
            },
        ) if dst_columns == src_columns && dst_rows == src_rows => Expression::As {
            expr: component,
            kind: ScalarKind::Float,
            convert: Some(dst_width),
        },

        // Matrix conversion (matrix -> matrix) - partial
        (
            Components::One {
                component,
                ty:
                    &TypeInner::Matrix {
                        columns: src_columns,
                        rows: src_rows,
                        ..
                    },
                ..
            },
            ConstructorType::PartialMatrix {
                columns: dst_columns,
                rows: dst_rows,
            },
        ) if dst_columns == src_columns && dst_rows == src_rows => Expression::As {
            expr: component,
            kind: ScalarKind::Float,
            convert: None,
        },

        // Vector constructor (splat) - infer type
        (
            Components::One {
                component,
                ty: &TypeInner::Scalar { .. },
                ..
            },
            ConstructorType::PartialVector { size },
        ) => Expression::Splat {
            size,
            value: component,
        },

        // Vector constructor (splat)
        (
            Components::One {
                component,
                ty:
                    &TypeInner::Scalar {
                        kind: src_kind,
                        width: src_width,
                        ..
                    },
                ..
            },
            ConstructorType::Vector {
                size,
                kind: dst_kind,
                width: dst_width,
            },
        ) if dst_kind == src_kind || dst_width == src_width => Expression::Splat {
            size,
            value: component,
        },

        // Vector constructor (by elements)
        (
            Components::Many {
                components,
                first_component_ty:
                    &TypeInner::Scalar { kind, width } | &TypeInner::Vector { kind, width, .. },
                ..
            },
            ConstructorType::PartialVector { size },
        )
        | (
            Components::Many {
                components,
                first_component_ty: &TypeInner::Scalar { .. } | &TypeInner::Vector { .. },
                ..
            },
            ConstructorType::Vector { size, width, kind },
        ) => {
            let ty = ctx.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector { size, kind, width },
                },
                Default::default(),
            );
            Expression::Compose { ty, components }
        }

        // Matrix constructor (by elements)
        (
            Components::Many {
                components,
                first_component_ty: &TypeInner::Scalar { width, .. },
                ..
            },
            ConstructorType::PartialMatrix { columns, rows },
        )
        | (
            Components::Many {
                components,
                first_component_ty: &TypeInner::Scalar { .. },
                ..
            },
            ConstructorType::Matrix {
                columns,
                rows,
                width,
            },
        ) => {
            let vec_ty = ctx.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector {
                        width,
                        kind: ScalarKind::Float,
                        size: rows,
                    },
                },
                Default::default(),
            );

            let components = components
                .chunks(rows as usize)
                .map(|vec_components| {
                    ctx.expressions.append(
                        Expression::Compose {
                            ty: vec_ty,
                            components: Vec::from(vec_components),
                        },
                        Default::default(),
                    )
                })
                .collect();

            let ty = ctx.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                },
                Default::default(),
            );
            Expression::Compose { ty, components }
        }

        // Matrix constructor (by columns)
        (
            Components::Many {
                components,
                first_component_ty: &TypeInner::Vector { width, .. },
                ..
            },
            ConstructorType::PartialMatrix { columns, rows },
        )
        | (
            Components::Many {
                components,
                first_component_ty: &TypeInner::Vector { .. },
                ..
            },
            ConstructorType::Matrix {
                columns,
                rows,
                width,
            },
        ) => {
            let ty = ctx.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                },
                Default::default(),
            );
            Expression::Compose { ty, components }
        }

        // Array constructor - infer type
        (components, ConstructorType::PartialArray) => {
            let components = components.into_components_vec();

            let base = match ctx.typifier[components[0]].clone() {
                TypeResolution::Handle(ty) => ty,
                TypeResolution::Value(inner) => ctx
                    .types
                    .insert(Type { name: None, inner }, Default::default()),
            };

            let size = Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Uint(components.len() as u64),
                },
            };

            let inner = TypeInner::Array {
                base,
                size: ArraySize::Constant(ctx.constants.append(size, Default::default())),
                stride: {
                    parser.layouter.update(ctx.types, ctx.constants).unwrap();
                    parser.layouter[base].to_stride()
                },
            };

            let ty = ctx
                .types
                .insert(Type { name: None, inner }, Default::default());

            Expression::Compose { ty, components }
        }

        // Array constructor
        (components, ConstructorType::Array { base, size, stride }) => {
            let components = components.into_components_vec();
            let inner = TypeInner::Array { base, size, stride };
            let ty = ctx
                .types
                .insert(Type { name: None, inner }, Default::default());
            Expression::Compose { ty, components }
        }

        // Struct constructor
        (components, ConstructorType::Struct(ty)) => Expression::Compose {
            ty,
            components: components.into_components_vec(),
        },

        // ERRORS

        // Bad conversion (type cast)
        (
            Components::One {
                span, ty: src_ty, ..
            },
            dst_ty,
        ) => {
            return Err(Error::BadTypeCast {
                span,
                from_type: src_ty.to_wgsl(ctx.types, ctx.constants),
                to_type: dst_ty.to_error_string(ctx.types, ctx.constants),
            });
        }

        // Too many parameters for scalar constructor
        (Components::Many { spans, .. }, ConstructorType::Scalar { .. }) => {
            return Err(Error::UnexpectedComponents(Span {
                start: spans[1].start,
                end: spans.last().unwrap().end,
            }));
        }

        // Parameters are of the wrong type for vector or matrix constructor
        (
            Components::Many { spans, .. },
            ConstructorType::Vector { .. }
            | ConstructorType::Matrix { .. }
            | ConstructorType::PartialVector { .. }
            | ConstructorType::PartialMatrix { .. },
        ) => {
            return Err(Error::InvalidConstructorComponentType(spans[0].clone(), 0));
        }
    };

    let span = NagaSpan::from(parser.pop_scope(lexer));
    Ok(Some(ctx.expressions.append(expr, span)))
}
