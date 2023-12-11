//! WGSL's automatic conversions for abstract types.

use crate::{Handle, Span};

impl<'source, 'temp, 'out> super::ExpressionContext<'source, 'temp, 'out> {
    /// Try to use WGSL's automatic conversions to convert `expr` to `goal_ty`.
    ///
    /// If no conversions are necessary, return `expr` unchanged.
    ///
    /// If automatic conversions cannot convert `expr` to `goal_ty`, return an
    /// [`AutoConversion`] error.
    ///
    /// Although the Load Rule is one of the automatic conversions, this
    /// function assumes it has already been applied if appropriate, as
    /// indicated by the fact that the Rust type of `expr` is not `Typed<_>`.
    ///
    /// [`AutoConversion`]: super::Error::AutoConversion
    pub fn try_automatic_conversions(
        &mut self,
        expr: Handle<crate::Expression>,
        goal_ty: &crate::proc::TypeResolution,
        goal_span: Span,
    ) -> Result<Handle<crate::Expression>, super::Error<'source>> {
        let expr_span = self.get_expression_span(expr);
        // Keep the TypeResolution so we can get type names for
        // structs in error messages.
        let expr_resolution = super::resolve!(self, expr);
        let types = &self.module.types;
        let expr_inner = expr_resolution.inner_with(types);
        let goal_inner = goal_ty.inner_with(types);

        // If `expr` already has the requested type, we're done.
        if expr_inner.equivalent(goal_inner, types) {
            return Ok(expr);
        }

        let (_expr_scalar, goal_scalar) =
            match expr_inner.automatically_converts_to(goal_inner, types) {
                Some(scalars) => scalars,
                None => {
                    let gctx = &self.module.to_ctx();
                    let source_type = expr_resolution.to_wgsl(gctx);
                    let dest_type = goal_ty.to_wgsl(gctx);

                    return Err(super::Error::AutoConversion {
                        dest_span: goal_span,
                        dest_type,
                        source_span: expr_span,
                        source_type,
                    });
                }
            };

        let converted = if let crate::TypeInner::Array { .. } = *goal_inner {
            let span = self.get_expression_span(expr);
            self.as_const_evaluator()
                .cast_array(expr, goal_scalar, span)
                .map_err(|err| super::Error::ConstantEvaluatorError(err, span))?
        } else {
            let cast = crate::Expression::As {
                expr,
                kind: goal_scalar.kind,
                convert: Some(goal_scalar.width),
            };
            self.append_expression(cast, expr_span)?
        };

        Ok(converted)
    }

    /// Try to convert `exprs` to `goal_ty` using WGSL's automatic conversions.
    pub fn try_automatic_conversions_slice(
        &mut self,
        exprs: &mut [Handle<crate::Expression>],
        goal_ty: &crate::proc::TypeResolution,
        goal_span: Span,
    ) -> Result<(), super::Error<'source>> {
        for expr in exprs.iter_mut() {
            *expr = self.try_automatic_conversions(*expr, goal_ty, goal_span)?;
        }

        Ok(())
    }

    /// Apply WGSL's automatic conversions to a vector constructor's arguments.
    ///
    /// When calling a vector constructor like `vec3<f32>(...)`, the parameters
    /// can be a mix of scalars and vectors, with the latter being spread out to
    /// contribute each of their components as a component of the new value.
    /// When the element type is explicit, as with `<f32>` in the example above,
    /// WGSL's automatic conversions should convert abstract scalar and vector
    /// parameters to the constructor's required scalar type.
    pub fn try_automatic_conversions_for_vector(
        &mut self,
        exprs: &mut [Handle<crate::Expression>],
        goal_scalar: crate::Scalar,
        goal_span: Span,
    ) -> Result<(), super::Error<'source>> {
        use crate::proc::TypeResolution as Tr;
        use crate::TypeInner as Ti;
        let goal_scalar_res = Tr::Value(Ti::Scalar(goal_scalar));

        for (i, expr) in exprs.iter_mut().enumerate() {
            // Keep the TypeResolution so we can get full type names
            // in error messages.
            let expr_resolution = super::resolve!(self, *expr);
            let types = &self.module.types;
            let expr_inner = expr_resolution.inner_with(types);

            match *expr_inner {
                Ti::Scalar(_) => {
                    *expr = self.try_automatic_conversions(*expr, &goal_scalar_res, goal_span)?;
                }
                Ti::Vector { size, scalar: _ } => {
                    let goal_vector_res = Tr::Value(Ti::Vector {
                        size,
                        scalar: goal_scalar,
                    });
                    *expr = self.try_automatic_conversions(*expr, &goal_vector_res, goal_span)?;
                }
                _ => {
                    let span = self.get_expression_span(*expr);
                    return Err(super::Error::InvalidConstructorComponentType(
                        span, i as i32,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Convert all expressions in `exprs` to a common scalar type.
    ///
    /// Note that the caller is responsible for making sure these
    /// conversions are actually justified. This function simply
    /// generates `As` expressions, regardless of whether they are
    /// permitted WGSL automatic conversions. Callers intending to
    /// implement automatic conversions need to determine for
    /// themselves whether the casts we we generate are justified,
    /// perhaps by calling `TypeInner::automatically_converts_to` or
    /// `Scalar::automatic_conversion_combine`.
    pub fn convert_slice_to_common_scalar(
        &mut self,
        exprs: &mut [Handle<crate::Expression>],
        goal: crate::Scalar,
    ) -> Result<(), super::Error<'source>> {
        for expr in exprs.iter_mut() {
            let inner = super::resolve_inner!(self, *expr);
            // Do nothing if `inner` doesn't even have leaf scalars;
            // it's a type error that validation will catch.
            if inner.scalar() != Some(goal) {
                let cast = crate::Expression::As {
                    expr: *expr,
                    kind: goal.kind,
                    convert: Some(goal.width),
                };
                let expr_span = self.get_expression_span(*expr);
                *expr = self.append_expression(cast, expr_span)?;
            }
        }

        Ok(())
    }

    /// Return an expression for the concretized value of `expr`.
    ///
    /// If `expr` is already concrete, return it unchanged.
    pub fn concretize(
        &mut self,
        mut expr: Handle<crate::Expression>,
    ) -> Result<Handle<crate::Expression>, super::Error<'source>> {
        let inner = super::resolve_inner!(self, expr);
        if let Some(scalar) = inner.automatically_convertible_scalar(&self.module.types) {
            let concretized = scalar.concretize();
            if concretized != scalar {
                assert!(scalar.is_abstract());
                let expr_span = self.get_expression_span(expr);
                expr = self
                    .as_const_evaluator()
                    .cast_array(expr, concretized, expr_span)
                    .map_err(|err| {
                        // A `TypeResolution` includes the type's full name, if
                        // it has one. Also, avoid holding the borrow of `inner`
                        // across the call to `cast_array`.
                        let expr_type = &self.typifier()[expr];
                        super::Error::ConcretizationFailed {
                            expr_span,
                            expr_type: expr_type.to_wgsl(&self.module.to_ctx()),
                            scalar: concretized.to_wgsl(),
                            inner: err,
                        }
                    })?;
            }
        }

        Ok(expr)
    }
}

impl crate::TypeInner {
    /// Determine whether `self` automatically converts to `goal`.
    ///
    /// If WGSL's automatic conversions (excluding the Load Rule) will
    /// convert `self` to `goal`, then return a pair `(from, to)`,
    /// where `from` and `to` are the scalar types of the leaf values
    /// of `self` and `goal`.
    ///
    /// This function assumes that `self` and `goal` are different
    /// types. Callers should first check whether any conversion is
    /// needed at all.
    ///
    /// If the automatic conversions cannot convert `self` to `goal`,
    /// return `None`.
    fn automatically_converts_to(
        &self,
        goal: &Self,
        types: &crate::UniqueArena<crate::Type>,
    ) -> Option<(crate::Scalar, crate::Scalar)> {
        use crate::ScalarKind as Sk;
        use crate::TypeInner as Ti;

        // Automatic conversions only change the scalar type of a value's leaves
        // (e.g., `vec4<AbstractFloat>` to `vec4<f32>`), never the type
        // constructors applied to those scalar types (e.g., never scalar to
        // `vec4`, or `vec2` to `vec3`). So first we check that the type
        // constructors match, extracting the leaf scalar types in the process.
        let expr_scalar;
        let goal_scalar;
        match (self, goal) {
            (&Ti::Scalar(expr), &Ti::Scalar(goal)) => {
                expr_scalar = expr;
                goal_scalar = goal;
            }
            (
                &Ti::Vector {
                    size: expr_size,
                    scalar: expr,
                },
                &Ti::Vector {
                    size: goal_size,
                    scalar: goal,
                },
            ) if expr_size == goal_size => {
                expr_scalar = expr;
                goal_scalar = goal;
            }
            (
                &Ti::Matrix {
                    rows: expr_rows,
                    columns: expr_columns,
                    scalar: expr,
                },
                &Ti::Matrix {
                    rows: goal_rows,
                    columns: goal_columns,
                    scalar: goal,
                },
            ) if expr_rows == goal_rows && expr_columns == goal_columns => {
                expr_scalar = expr;
                goal_scalar = goal;
            }
            (
                &Ti::Array {
                    base: expr_base,
                    size: expr_size,
                    stride: _,
                },
                &Ti::Array {
                    base: goal_base,
                    size: goal_size,
                    stride: _,
                },
            ) if expr_size == goal_size => {
                return types[expr_base]
                    .inner
                    .automatically_converts_to(&types[goal_base].inner, types);
            }
            _ => return None,
        }

        match (expr_scalar.kind, goal_scalar.kind) {
            (Sk::AbstractFloat, Sk::Float) => {}
            (Sk::AbstractInt, Sk::Sint | Sk::Uint | Sk::AbstractFloat | Sk::Float) => {}
            _ => return None,
        }

        log::trace!("      okay: expr {expr_scalar:?}, goal {goal_scalar:?}");
        Some((expr_scalar, goal_scalar))
    }

    fn automatically_convertible_scalar(
        &self,
        types: &crate::UniqueArena<crate::Type>,
    ) -> Option<crate::Scalar> {
        use crate::TypeInner as Ti;
        match *self {
            Ti::Scalar(scalar) | Ti::Vector { scalar, .. } | Ti::Matrix { scalar, .. } => {
                Some(scalar)
            }
            Ti::Array { base, .. } => types[base].inner.automatically_convertible_scalar(types),
            Ti::Atomic(_)
            | Ti::Pointer { .. }
            | Ti::ValuePointer { .. }
            | Ti::Struct { .. }
            | Ti::Image { .. }
            | Ti::Sampler { .. }
            | Ti::AccelerationStructure
            | Ti::RayQuery
            | Ti::BindingArray { .. } => None,
        }
    }
}

impl crate::Scalar {
    /// Find the common type of `self` and `other` under WGSL's
    /// automatic conversions.
    ///
    /// If there are any scalars to which WGSL's automatic conversions
    /// will convert both `self` and `other`, return the best such
    /// scalar. Otherwise, return `None`.
    pub const fn automatic_conversion_combine(self, other: Self) -> Option<crate::Scalar> {
        use crate::ScalarKind as Sk;

        match (self.kind, other.kind) {
            // When the kinds match...
            (Sk::AbstractFloat, Sk::AbstractFloat)
            | (Sk::AbstractInt, Sk::AbstractInt)
            | (Sk::Sint, Sk::Sint)
            | (Sk::Uint, Sk::Uint)
            | (Sk::Float, Sk::Float)
            | (Sk::Bool, Sk::Bool) => {
                if self.width == other.width {
                    // ... either no conversion is necessary ...
                    Some(self)
                } else {
                    // ... or no conversion is possible.
                    // We never convert concrete to concrete, and
                    // abstract types should have only one size.
                    None
                }
            }

            // AbstractInt converts to AbstractFloat.
            (Sk::AbstractFloat, Sk::AbstractInt) => Some(self),
            (Sk::AbstractInt, Sk::AbstractFloat) => Some(other),

            // AbstractFloat converts to Float.
            (Sk::AbstractFloat, Sk::Float) => Some(other),
            (Sk::Float, Sk::AbstractFloat) => Some(self),

            // AbstractInt converts to concrete integer or float.
            (Sk::AbstractInt, Sk::Uint | Sk::Sint | Sk::Float) => Some(other),
            (Sk::Uint | Sk::Sint | Sk::Float, Sk::AbstractInt) => Some(self),

            // AbstractFloat can't be reconciled with concrete integer types.
            (Sk::AbstractFloat, Sk::Uint | Sk::Sint) | (Sk::Uint | Sk::Sint, Sk::AbstractFloat) => {
                None
            }

            // Nothing can be reconciled with `bool`.
            (Sk::Bool, _) | (_, Sk::Bool) => None,

            // Different concrete types cannot be reconciled.
            (Sk::Sint | Sk::Uint | Sk::Float, Sk::Sint | Sk::Uint | Sk::Float) => None,
        }
    }

    const fn concretize(self) -> Self {
        use crate::ScalarKind as Sk;
        match self.kind {
            Sk::Sint | Sk::Uint | Sk::Float | Sk::Bool => self,
            Sk::AbstractInt => Self::I32,
            Sk::AbstractFloat => Self::F32,
        }
    }
}
