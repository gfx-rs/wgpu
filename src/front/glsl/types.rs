use super::{context::Context, Error, ErrorKind, Result, Span};
use crate::{
    proc::ResolveContext, Bytes, Expression, Handle, ImageClass, ImageDimension, ScalarKind, Type,
    TypeInner, VectorSize,
};

pub fn parse_type(type_name: &str) -> Option<Type> {
    match type_name {
        "bool" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Bool,
                width: crate::BOOL_WIDTH,
            },
        }),
        "float" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            },
        }),
        "double" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 8,
            },
        }),
        "int" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            },
        }),
        "uint" => Some(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
        }),
        "sampler" | "samplerShadow" => Some(Type {
            name: None,
            inner: TypeInner::Sampler {
                comparison: type_name == "samplerShadow",
            },
        }),
        word => {
            fn kind_width_parse(ty: &str) -> Option<(ScalarKind, u8)> {
                Some(match ty {
                    "" => (ScalarKind::Float, 4),
                    "b" => (ScalarKind::Bool, crate::BOOL_WIDTH),
                    "i" => (ScalarKind::Sint, 4),
                    "u" => (ScalarKind::Uint, 4),
                    "d" => (ScalarKind::Float, 8),
                    _ => return None,
                })
            }

            fn size_parse(n: &str) -> Option<VectorSize> {
                Some(match n {
                    "2" => VectorSize::Bi,
                    "3" => VectorSize::Tri,
                    "4" => VectorSize::Quad,
                    _ => return None,
                })
            }

            let vec_parse = |word: &str| {
                let mut iter = word.split("vec");

                let kind = iter.next()?;
                let size = iter.next()?;
                let (kind, width) = kind_width_parse(kind)?;
                let size = size_parse(size)?;

                Some(Type {
                    name: None,
                    inner: TypeInner::Vector { size, kind, width },
                })
            };

            let mat_parse = |word: &str| {
                let mut iter = word.split("mat");

                let kind = iter.next()?;
                let size = iter.next()?;
                let (_, width) = kind_width_parse(kind)?;

                let (columns, rows) = if let Some(size) = size_parse(size) {
                    (size, size)
                } else {
                    let mut iter = size.split('x');
                    match (iter.next()?, iter.next()?, iter.next()) {
                        (col, row, None) => (size_parse(col)?, size_parse(row)?),
                        _ => return None,
                    }
                };

                Some(Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                })
            };

            let texture_parse = |word: &str| {
                let mut iter = word.split("texture");

                let texture_kind = |ty| {
                    Some(match ty {
                        "" => ScalarKind::Float,
                        "i" => ScalarKind::Sint,
                        "u" => ScalarKind::Uint,
                        _ => return None,
                    })
                };

                let kind = iter.next()?;
                let size = iter.next()?;
                let kind = texture_kind(kind)?;

                let sampled = |multi| ImageClass::Sampled { kind, multi };

                let (dim, arrayed, class) = match size {
                    "1D" => (ImageDimension::D1, false, sampled(false)),
                    "1DArray" => (ImageDimension::D1, true, sampled(false)),
                    "2D" => (ImageDimension::D2, false, sampled(false)),
                    "2DArray" => (ImageDimension::D2, true, sampled(false)),
                    "2DMS" => (ImageDimension::D2, false, sampled(true)),
                    "2DMSArray" => (ImageDimension::D2, true, sampled(true)),
                    "3D" => (ImageDimension::D3, false, sampled(false)),
                    "Cube" => (ImageDimension::Cube, false, sampled(false)),
                    "CubeArray" => (ImageDimension::Cube, true, sampled(false)),
                    _ => return None,
                };

                Some(Type {
                    name: None,
                    inner: TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    },
                })
            };

            let image_parse = |word: &str| {
                let mut iter = word.split("image");

                let texture_kind = |ty| {
                    Some(match ty {
                        "" => ScalarKind::Float,
                        "i" => ScalarKind::Sint,
                        "u" => ScalarKind::Uint,
                        _ => return None,
                    })
                };

                let kind = iter.next()?;
                let size = iter.next()?;
                // TODO: Check that the texture format and the kind match
                let _ = texture_kind(kind)?;

                let class = ImageClass::Storage {
                    format: crate::StorageFormat::R8Uint,
                    access: crate::StorageAccess::all(),
                };

                // TODO: glsl support multisampled storage images, naga doesn't
                let (dim, arrayed) = match size {
                    "1D" => (ImageDimension::D1, false),
                    "1DArray" => (ImageDimension::D1, true),
                    "2D" => (ImageDimension::D2, false),
                    "2DArray" => (ImageDimension::D2, true),
                    "3D" => (ImageDimension::D3, false),
                    // Naga doesn't support cube images and it's usefulness
                    // is questionable, so they won't be supported for now
                    // "Cube" => (ImageDimension::Cube, false),
                    // "CubeArray" => (ImageDimension::Cube, true),
                    _ => return None,
                };

                Some(Type {
                    name: None,
                    inner: TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    },
                })
            };

            vec_parse(word)
                .or_else(|| mat_parse(word))
                .or_else(|| texture_parse(word))
                .or_else(|| image_parse(word))
        }
    }
}

pub const fn scalar_components(ty: &TypeInner) -> Option<(ScalarKind, Bytes)> {
    match *ty {
        TypeInner::Scalar { kind, width } => Some((kind, width)),
        TypeInner::Vector { kind, width, .. } => Some((kind, width)),
        TypeInner::Matrix { width, .. } => Some((ScalarKind::Float, width)),
        TypeInner::ValuePointer { kind, width, .. } => Some((kind, width)),
        _ => None,
    }
}

pub const fn type_power(kind: ScalarKind, width: Bytes) -> Option<u32> {
    Some(match kind {
        ScalarKind::Sint => 0,
        ScalarKind::Uint => 1,
        ScalarKind::Float if width == 4 => 2,
        ScalarKind::Float => 3,
        ScalarKind::Bool => return None,
    })
}

impl Context<'_> {
    /// Resolves the types of the expressions until `expr` (inclusive)
    ///
    /// This needs to be done before the [`typifier`] can be queried for
    /// the types of the expressions in the range between the last grow and `expr`.
    ///
    /// # Note
    ///
    /// The `resolve_type*` methods (like [`resolve_type`]) automatically
    /// grow the [`typifier`] so calling this method is not necessary when using
    /// them.
    ///
    /// [`typifier`]: Context::typifier
    /// [`resolve_type`]: Self::resolve_type
    pub(crate) fn typifier_grow(&mut self, expr: Handle<Expression>, meta: Span) -> Result<()> {
        let resolve_ctx = ResolveContext::with_locals(self.module, &self.locals, &self.arguments);

        self.typifier
            .grow(expr, &self.expressions, &resolve_ctx)
            .map_err(|error| Error {
                kind: ErrorKind::SemanticError(format!("Can't resolve type: {error:?}").into()),
                meta,
            })
    }

    /// Gets the type for the result of the `expr` expression
    ///
    /// Automatically grows the [`typifier`] to `expr` so calling
    /// [`typifier_grow`] is not necessary
    ///
    /// [`typifier`]: Context::typifier
    /// [`typifier_grow`]: Self::typifier_grow
    pub(crate) fn resolve_type(
        &mut self,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<&TypeInner> {
        self.typifier_grow(expr, meta)?;
        Ok(self.typifier.get(expr, &self.module.types))
    }

    /// Gets the type handle for the result of the `expr` expression
    ///
    /// Automatically grows the [`typifier`] to `expr` so calling
    /// [`typifier_grow`] is not necessary
    ///
    /// # Note
    ///
    /// Consider using [`resolve_type`] whenever possible
    /// since it doesn't require adding each type to the [`types`] arena
    /// and it doesn't need to mutably borrow the [`Parser`][Self]
    ///
    /// [`types`]: crate::Module::types
    /// [`typifier`]: Context::typifier
    /// [`typifier_grow`]: Self::typifier_grow
    /// [`resolve_type`]: Self::resolve_type
    pub(crate) fn resolve_type_handle(
        &mut self,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<Handle<Type>> {
        self.typifier_grow(expr, meta)?;
        Ok(self.typifier.register_type(expr, &mut self.module.types))
    }

    /// Invalidates the cached type resolution for `expr` forcing a recomputation
    pub(crate) fn invalidate_expression(
        &mut self,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<()> {
        let resolve_ctx = ResolveContext::with_locals(self.module, &self.locals, &self.arguments);

        self.typifier
            .invalidate(expr, &self.expressions, &resolve_ctx)
            .map_err(|error| Error {
                kind: ErrorKind::SemanticError(format!("Can't resolve type: {error:?}").into()),
                meta,
            })
    }

    pub(crate) fn eval_constant(
        &mut self,
        root: Handle<Expression>,
        meta: Span,
    ) -> Result<Handle<Expression>> {
        let mut solver = crate::proc::ConstantEvaluator {
            types: &mut self.module.types,
            expressions: &mut self.module.const_expressions,
            constants: &mut self.module.constants,
            const_expressions: Some(&self.expressions),
            append: None::<
                Box<
                    dyn FnMut(
                        &mut crate::Arena<Expression>,
                        Expression,
                        Span,
                    ) -> Handle<Expression>,
                >,
            >,
        };

        solver.eval(root).map_err(|e| Error {
            kind: e.into(),
            meta,
        })
    }
}
