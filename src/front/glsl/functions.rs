use super::{ast::*, error::ErrorKind, SourceMetadata};
use crate::{
    proc::ensure_block_returns, Arena, BinaryOperator, Block, Constant, ConstantInner, EntryPoint,
    Expression, Function, FunctionArgument, FunctionResult, Handle, ImageClass, ImageDimension,
    ImageQuery, LocalVariable, MathFunction, Module, RelationalFunction, SampleLevel, ScalarKind,
    ScalarValue, Statement, StructMember, Type, TypeInner, VectorSize,
};

/// Helper struct for texture calls with the separate components from the vector argument
///
/// Obtained by calling [`coordinate_components`](Program::coordinate_components)
struct CoordComponents {
    coordinate: Handle<Expression>,
    depth_ref: Option<Handle<Expression>>,
    array_index: Option<Handle<Expression>>,
}

impl Program {
    fn add_constant_value(&mut self, scalar_kind: ScalarKind, value: u64) -> Handle<Constant> {
        let value = match scalar_kind {
            ScalarKind::Uint => ScalarValue::Uint(value),
            ScalarKind::Sint => ScalarValue::Sint(value as i64),
            ScalarKind::Float => ScalarValue::Float(value as f64),
            _ => unreachable!(),
        };

        self.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar { width: 4, value },
        })
    }

    pub fn function_or_constructor_call(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        fc: FunctionCallKind,
        raw_args: &[Handle<HirExpr>],
        meta: SourceMetadata,
    ) -> Result<Option<Handle<Expression>>, ErrorKind> {
        let args: Vec<_> = raw_args
            .iter()
            .map(|e| ctx.lower_expect(self, *e, false, body))
            .collect::<Result<_, _>>()?;

        match fc {
            FunctionCallKind::TypeConstructor(ty) => {
                let h = if args.len() == 1 {
                    let expr_type = self.resolve_type(ctx, args[0].0, args[0].1)?;

                    let vector_size = match *expr_type {
                        TypeInner::Vector { size, .. } => Some(size),
                        _ => None,
                    };

                    // Special case: if casting from a bool, we need to use Select and not As.
                    match self.module.types[ty].inner.scalar_kind() {
                        Some(result_scalar_kind)
                            if expr_type.scalar_kind() == Some(ScalarKind::Bool)
                                && result_scalar_kind != ScalarKind::Bool =>
                        {
                            let c0 = self.add_constant_value(result_scalar_kind, 0u64);
                            let c1 = self.add_constant_value(result_scalar_kind, 1u64);
                            let mut reject = ctx.add_expression(Expression::Constant(c0), body);
                            let mut accept = ctx.add_expression(Expression::Constant(c1), body);

                            ctx.implicit_splat(self, &mut reject, meta, vector_size)?;
                            ctx.implicit_splat(self, &mut accept, meta, vector_size)?;

                            let h = ctx.add_expression(
                                Expression::Select {
                                    accept,
                                    reject,
                                    condition: args[0].0,
                                },
                                body,
                            );

                            return Ok(Some(h));
                        }
                        _ => {}
                    }

                    match self.module.types[ty].inner {
                        TypeInner::Vector { size, kind, width } if vector_size.is_none() => {
                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(self, &mut value, meta, kind, width)?;

                            ctx.add_expression(Expression::Splat { size, value }, body)
                        }
                        TypeInner::Scalar { kind, width } => ctx.add_expression(
                            Expression::As {
                                kind,
                                expr: args[0].0,
                                convert: Some(width),
                            },
                            body,
                        ),
                        TypeInner::Vector { size, kind, width } => {
                            let mut expr = args[0].0;

                            if vector_size.map_or(true, |s| s != size) {
                                expr = ctx.vector_resize(size, expr, body);
                            }

                            ctx.add_expression(
                                Expression::As {
                                    kind,
                                    expr,
                                    convert: Some(width),
                                },
                                body,
                            )
                        }
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            // TODO: casts
                            // `Expression::As` doesn't support matrix width
                            // casts so we need to do some extra work for casts

                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(
                                self,
                                &mut value,
                                meta,
                                ScalarKind::Float,
                                width,
                            )?;
                            let column = match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                                TypeInner::Scalar { .. } => ctx
                                    .add_expression(Expression::Splat { size: rows, value }, body),
                                TypeInner::Matrix { rows: ori_rows, .. } => {
                                    let mut components = Vec::new();

                                    for n in 0..columns as u32 {
                                        let mut vector = ctx.add_expression(
                                            Expression::AccessIndex {
                                                base: value,
                                                index: n,
                                            },
                                            body,
                                        );

                                        if ori_rows != rows {
                                            vector = ctx.vector_resize(rows, vector, body);
                                        }

                                        components.push(vector)
                                    }

                                    let h = ctx.add_expression(
                                        Expression::Compose { ty, components },
                                        body,
                                    );

                                    return Ok(Some(h));
                                }
                                _ => value,
                            };

                            let columns =
                                std::iter::repeat(column).take(columns as usize).collect();

                            ctx.add_expression(
                                Expression::Compose {
                                    ty,
                                    components: columns,
                                },
                                body,
                            )
                        }
                        TypeInner::Struct { .. } => ctx.add_expression(
                            Expression::Compose {
                                ty,
                                components: args.into_iter().map(|arg| arg.0).collect(),
                            },
                            body,
                        ),
                        _ => return Err(ErrorKind::SemanticError(meta, "Bad cast".into())),
                    }
                } else {
                    let mut components = Vec::with_capacity(args.len());

                    match self.module.types[ty].inner {
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            let mut flattened =
                                Vec::with_capacity(columns as usize * rows as usize);

                            for (mut arg, meta) in args.iter().copied() {
                                let scalar_components =
                                    scalar_components(&self.module.types[ty].inner);
                                if let Some((kind, width)) = scalar_components {
                                    ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                                }

                                match *self.resolve_type(ctx, arg, meta)? {
                                    TypeInner::Vector { size, .. } => {
                                        for i in 0..(size as u32) {
                                            flattened.push(ctx.add_expression(
                                                Expression::AccessIndex {
                                                    base: arg,
                                                    index: i,
                                                },
                                                body,
                                            ))
                                        }
                                    }
                                    _ => flattened.push(arg),
                                }
                            }

                            let ty = self.module.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Vector {
                                    size: rows,
                                    kind: ScalarKind::Float,
                                    width,
                                },
                            });

                            for chunk in flattened.chunks(rows as usize) {
                                components.push(ctx.add_expression(
                                    Expression::Compose {
                                        ty,
                                        components: Vec::from(chunk),
                                    },
                                    body,
                                ))
                            }
                        }
                        _ => {
                            for (mut arg, meta) in args.iter().copied() {
                                let scalar_components =
                                    scalar_components(&self.module.types[ty].inner);
                                if let Some((kind, width)) = scalar_components {
                                    ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                                }

                                components.push(arg)
                            }
                        }
                    }

                    ctx.add_expression(Expression::Compose { ty, components }, body)
                };

                Ok(Some(h))
            }
            FunctionCallKind::Function(name) => {
                self.function_call(ctx, body, name, args, raw_args, meta)
            }
        }
    }

    fn function_call(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        name: String,
        mut args: Vec<(Handle<Expression>, SourceMetadata)>,
        raw_args: &[Handle<HirExpr>],
        meta: SourceMetadata,
    ) -> Result<Option<Handle<Expression>>, ErrorKind> {
        match name.as_str() {
            "sampler1D" | "sampler1DArray" | "sampler2D" | "sampler2DArray" | "sampler2DMS"
            | "sampler2DMSArray" | "sampler3D" | "samplerCube" | "samplerCubeArray" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }
                ctx.samplers.insert(args[0].0, args[1].0);
                Ok(Some(args[0].0))
            }
            "sampler1DShadow"
            | "sampler1DArrayShadow"
            | "sampler2DShadow"
            | "sampler2DArrayShadow"
            | "samplerCubeShadow"
            | "samplerCubeArrayShadow" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }
                sampled_to_depth(&mut self.module, ctx, args[0])?;
                self.invalidate_expression(ctx, args[0].0, args[0].1)?;
                ctx.samplers.insert(args[0].0, args[1].0);
                Ok(Some(args[0].0))
            }
            "texture" => {
                if !(2..=3).contains(&args.len()) {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }
                let arg_1 = &mut args[1];
                ctx.implicit_conversion(self, &mut arg_1.0, arg_1.1, ScalarKind::Float, 4)?;
                if let Some(&mut (ref mut expr, meta)) = args.get_mut(2) {
                    ctx.implicit_conversion(self, expr, meta, ScalarKind::Float, 4)?;
                }
                let comps = self.coordinate_components(ctx, args[0], args[1], body)?;
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(
                        ctx.add_expression(
                            Expression::ImageSample {
                                image: args[0].0,
                                sampler,
                                coordinate: comps.coordinate,
                                array_index: comps.array_index,
                                offset: None,
                                level: args.get(2).map_or(SampleLevel::Auto, |&(expr, _)| {
                                    SampleLevel::Bias(expr)
                                }),
                                depth_ref: comps.depth_ref,
                            },
                            body,
                        ),
                    ))
                } else {
                    Err(ErrorKind::SemanticError(meta, "Bad call to texture".into()))
                }
            }
            "textureLod" => {
                if args.len() != 3 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }
                let arg_1 = &mut args[1];
                ctx.implicit_conversion(self, &mut arg_1.0, arg_1.1, ScalarKind::Float, 4)?;
                let arg_2 = &mut args[2];
                ctx.implicit_conversion(self, &mut arg_2.0, arg_2.1, ScalarKind::Float, 4)?;
                let comps = self.coordinate_components(ctx, args[0], args[1], body)?;
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: comps.coordinate,
                            array_index: comps.array_index,
                            offset: None,
                            level: SampleLevel::Exact(args[2].0),
                            depth_ref: comps.depth_ref,
                        },
                        body,
                    )))
                } else {
                    Err(ErrorKind::SemanticError(
                        meta,
                        "Bad call to textureLod".into(),
                    ))
                }
            }
            "textureProj" => {
                if !(2..=3).contains(&args.len()) {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }
                let arg_1 = &mut args[1];
                ctx.implicit_conversion(self, &mut arg_1.0, arg_1.1, ScalarKind::Float, 4)?;
                if let Some(&mut (ref mut expr, meta)) = args.get_mut(2) {
                    ctx.implicit_conversion(self, expr, meta, ScalarKind::Float, 4)?;
                }
                let level = args
                    .get(2)
                    .map_or(SampleLevel::Auto, |&(expr, _)| SampleLevel::Bias(expr));
                let size = match *self.resolve_type(ctx, args[1].0, args[1].1)? {
                    TypeInner::Vector { size, .. } => size,
                    _ => {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Bad call to textureProj".into(),
                        ))
                    }
                };
                let (base, base_meta) = args[1];
                let mut right = ctx.add_expression(
                    Expression::AccessIndex {
                        base,
                        index: size as u32 - 1,
                    },
                    body,
                );
                let left = if let VectorSize::Bi = size {
                    ctx.add_expression(Expression::AccessIndex { base, index: 0 }, body)
                } else {
                    let size = match size {
                        VectorSize::Tri => VectorSize::Bi,
                        _ => VectorSize::Tri,
                    };
                    right = ctx.add_expression(Expression::Splat { size, value: right }, body);
                    ctx.vector_resize(size, base, body)
                };
                let coords = ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left,
                        right,
                    },
                    body,
                );
                let comps = self.coordinate_components(ctx, args[0], (coords, base_meta), body)?;
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: comps.coordinate,
                            array_index: comps.array_index,
                            offset: None,
                            level,
                            depth_ref: comps.depth_ref,
                        },
                        body,
                    )))
                } else {
                    Err(ErrorKind::SemanticError(
                        meta,
                        "Bad call to textureProj".into(),
                    ))
                }
            }
            "textureGrad" => {
                if args.len() != 4 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }
                let arg_1 = &mut args[1];
                ctx.implicit_conversion(self, &mut arg_1.0, arg_1.1, ScalarKind::Float, 4)?;
                let arg_2 = &mut args[2];
                ctx.implicit_conversion(self, &mut arg_2.0, arg_2.1, ScalarKind::Float, 4)?;
                let arg_3 = &mut args[3];
                ctx.implicit_conversion(self, &mut arg_3.0, arg_3.1, ScalarKind::Float, 4)?;
                let comps = self.coordinate_components(ctx, args[0], args[1], body)?;
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: comps.coordinate,
                            array_index: comps.array_index,
                            offset: None,
                            level: SampleLevel::Gradient {
                                x: args[2].0,
                                y: args[3].0,
                            },
                            depth_ref: comps.depth_ref,
                        },
                        body,
                    )))
                } else {
                    Err(ErrorKind::SemanticError(
                        meta,
                        "Bad call to textureGrad".into(),
                    ))
                }
            }
            "textureSize" => {
                if !(1..=2).contains(&args.len()) {
                    return Err(ErrorKind::wrong_function_args(name, 1, args.len(), meta));
                }
                if let Some(&mut (ref mut expr, meta)) = args.get_mut(1) {
                    ctx.implicit_conversion(self, expr, meta, ScalarKind::Sint, 4)?;
                }
                Ok(Some(ctx.add_expression(
                    Expression::ImageQuery {
                        image: args[0].0,
                        query: ImageQuery::Size {
                            level: args.get(1).map(|e| e.0),
                        },
                    },
                    body,
                )))
            }
            "texelFetch" => {
                if args.len() != 3 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }
                let arg_1 = &mut args[1];
                ctx.implicit_conversion(self, &mut arg_1.0, arg_1.1, ScalarKind::Sint, 4)?;
                let arg_2 = &mut args[2];
                ctx.implicit_conversion(self, &mut arg_2.0, arg_2.1, ScalarKind::Sint, 4)?;
                let comps = self.coordinate_components(ctx, args[0], args[1], body)?;
                if ctx.samplers.get(&args[0].0).is_some() && comps.depth_ref.is_none() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageLoad {
                            image: args[0].0,
                            coordinate: comps.coordinate,
                            array_index: comps.array_index,
                            index: Some(args[2].0),
                        },
                        body,
                    )))
                } else {
                    Err(ErrorKind::SemanticError(
                        meta,
                        "Bad call to texelFetch".into(),
                    ))
                }
            }
            "ceil" | "round" | "floor" | "fract" | "trunc" | "sin" | "abs" | "sqrt"
            | "inversesqrt" | "exp" | "exp2" | "sign" | "transpose" | "inverse" | "normalize"
            | "sinh" | "cos" | "cosh" | "tan" | "tanh" | "acos" | "asin" | "log" | "log2"
            | "length" | "determinant" | "bitCount" | "bitfieldReverse" => {
                if args.len() != 1 {
                    return Err(ErrorKind::wrong_function_args(name, 1, args.len(), meta));
                }
                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: match name.as_str() {
                            "ceil" => MathFunction::Ceil,
                            "round" => MathFunction::Round,
                            "floor" => MathFunction::Floor,
                            "fract" => MathFunction::Fract,
                            "trunc" => MathFunction::Trunc,
                            "sin" => MathFunction::Sin,
                            "abs" => MathFunction::Abs,
                            "sqrt" => MathFunction::Sqrt,
                            "inversesqrt" => MathFunction::InverseSqrt,
                            "exp" => MathFunction::Exp,
                            "exp2" => MathFunction::Exp2,
                            "sign" => MathFunction::Sign,
                            "transpose" => MathFunction::Transpose,
                            "inverse" => MathFunction::Inverse,
                            "normalize" => MathFunction::Normalize,
                            "sinh" => MathFunction::Sinh,
                            "cos" => MathFunction::Cos,
                            "cosh" => MathFunction::Cosh,
                            "tan" => MathFunction::Tan,
                            "tanh" => MathFunction::Tanh,
                            "acos" => MathFunction::Acos,
                            "asin" => MathFunction::Asin,
                            "log" => MathFunction::Log,
                            "log2" => MathFunction::Log2,
                            "length" => MathFunction::Length,
                            "determinant" => MathFunction::Determinant,
                            "bitCount" => MathFunction::CountOneBits,
                            "bitfieldReverse" => MathFunction::ReverseBits,
                            _ => unreachable!(),
                        },
                        arg: args[0].0,
                        arg1: None,
                        arg2: None,
                    },
                    body,
                )))
            }
            "atan" => {
                let expr = match args.len() {
                    1 => Expression::Math {
                        fun: MathFunction::Atan,
                        arg: args[0].0,
                        arg1: None,
                        arg2: None,
                    },
                    2 => Expression::Math {
                        fun: MathFunction::Atan2,
                        arg: args[0].0,
                        arg1: Some(args[1].0),
                        arg2: None,
                    },
                    _ => return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta)),
                };
                Ok(Some(ctx.add_expression(expr, body)))
            }
            "mod" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }

                let (mut left, left_meta) = args[0];
                let (mut right, right_meta) = args[1];

                ctx.binary_implicit_conversion(self, &mut left, left_meta, &mut right, right_meta)?;

                Ok(Some(ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left,
                        right,
                    },
                    body,
                )))
            }
            "min" | "max" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }

                let (mut arg0, arg0_meta) = args[0];
                let (mut arg1, arg1_meta) = args[1];

                ctx.binary_implicit_conversion(self, &mut arg0, arg0_meta, &mut arg1, arg1_meta)?;

                if let TypeInner::Vector { size, .. } = *self.resolve_type(ctx, arg0, arg0_meta)? {
                    ctx.implicit_splat(self, &mut arg1, arg1_meta, Some(size))?;
                }
                if let TypeInner::Vector { size, .. } = *self.resolve_type(ctx, arg1, arg1_meta)? {
                    ctx.implicit_splat(self, &mut arg0, arg0_meta, Some(size))?;
                }

                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: match name.as_str() {
                            "min" => MathFunction::Min,
                            "max" => MathFunction::Max,
                            _ => unreachable!(),
                        },
                        arg: arg0,
                        arg1: Some(arg1),
                        arg2: None,
                    },
                    body,
                )))
            }
            "pow" | "dot" | "reflect" | "cross" | "outerProduct" | "distance" | "step" | "modf"
            | "frexp" | "ldexp" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }

                let (mut arg0, arg0_meta) = args[0];
                let (mut arg1, arg1_meta) = args[1];

                ctx.binary_implicit_conversion(self, &mut arg0, arg0_meta, &mut arg1, arg1_meta)?;

                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: match name.as_str() {
                            "pow" => MathFunction::Pow,
                            "dot" => MathFunction::Dot,
                            "reflect" => MathFunction::Reflect,
                            "cross" => MathFunction::Cross,
                            "outerProduct" => MathFunction::Outer,
                            "distance" => MathFunction::Distance,
                            "step" => MathFunction::Step,
                            "modf" => MathFunction::Modf,
                            "frexp" => MathFunction::Frexp,
                            "ldexp" => MathFunction::Ldexp,
                            _ => unreachable!(),
                        },
                        arg: arg0,
                        arg1: Some(arg1),
                        arg2: None,
                    },
                    body,
                )))
            }
            "mix" => {
                if args.len() != 3 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }

                let (mut arg, arg_meta) = args[0];
                let (mut arg1, arg1_meta) = args[1];
                let (mut selector, selector_meta) = args[2];

                ctx.binary_implicit_conversion(self, &mut arg, arg_meta, &mut arg1, arg1_meta)?;
                ctx.binary_implicit_conversion(
                    self,
                    &mut arg,
                    arg_meta,
                    &mut selector,
                    selector_meta,
                )?;
                ctx.binary_implicit_conversion(
                    self,
                    &mut arg1,
                    arg1_meta,
                    &mut selector,
                    selector_meta,
                )?;

                let is_vector = match *self.resolve_type(ctx, selector, selector_meta)? {
                    TypeInner::Vector { .. } => true,
                    _ => false,
                };
                match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                    TypeInner::Vector { size, .. } if !is_vector => {
                        selector = ctx.add_expression(
                            Expression::Splat {
                                size,
                                value: selector,
                            },
                            body,
                        )
                    }
                    _ => {}
                };

                let expr = match self
                    .resolve_type(ctx, selector, selector_meta)?
                    .scalar_kind()
                {
                    Some(ScalarKind::Bool) => Expression::Select {
                        condition: selector,
                        accept: arg,
                        reject: arg1,
                    },
                    _ => Expression::Math {
                        fun: MathFunction::Mix,
                        arg,
                        arg1: Some(arg1),
                        arg2: Some(selector),
                    },
                };

                Ok(Some(ctx.add_expression(expr, body)))
            }
            "clamp" => {
                if args.len() != 3 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }

                let (mut arg0, arg0_meta) = args[0];
                let (mut arg1, arg1_meta) = args[1];
                let (mut arg2, arg2_meta) = args[2];

                let vector_size = match *(self.resolve_type(ctx, arg0, arg0_meta)?) {
                    TypeInner::Vector { size, .. } => Some(size),
                    _ => None,
                };

                ctx.binary_implicit_conversion(self, &mut arg0, arg0_meta, &mut arg1, arg1_meta)?;
                ctx.binary_implicit_conversion(self, &mut arg1, arg1_meta, &mut arg2, arg2_meta)?;
                ctx.binary_implicit_conversion(self, &mut arg2, arg2_meta, &mut arg0, arg0_meta)?;

                ctx.implicit_splat(self, &mut arg1, arg1_meta, vector_size)?;
                ctx.implicit_splat(self, &mut arg2, arg2_meta, vector_size)?;

                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: arg0,
                        arg1: Some(arg1),
                        arg2: Some(arg2),
                    },
                    body,
                )))
            }
            "faceforward" | "refract" | "fma" | "smoothstep" => {
                if args.len() != 3 {
                    return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                }
                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: match name.as_str() {
                            "faceforward" => MathFunction::FaceForward,
                            "refract" => MathFunction::Refract,
                            "fma" => MathFunction::Fma,
                            "smoothstep" => MathFunction::SmoothStep,
                            _ => unreachable!(),
                        },
                        arg: args[0].0,
                        arg1: Some(args[1].0),
                        arg2: Some(args[2].0),
                    },
                    body,
                )))
            }
            "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
            | "notEqual" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }
                Ok(Some(ctx.add_expression(
                    Expression::Binary {
                        op: match name.as_str() {
                            "lessThan" => BinaryOperator::Less,
                            "greaterThan" => BinaryOperator::Greater,
                            "lessThanEqual" => BinaryOperator::LessEqual,
                            "greaterThanEqual" => BinaryOperator::GreaterEqual,
                            "equal" => BinaryOperator::Equal,
                            "notEqual" => BinaryOperator::NotEqual,
                            _ => unreachable!(),
                        },
                        left: args[0].0,
                        right: args[1].0,
                    },
                    body,
                )))
            }
            "isinf" | "isnan" | "all" | "any" => {
                let fun = match name.as_str() {
                    "isinf" => RelationalFunction::IsInf,
                    "isnan" => RelationalFunction::IsNan,
                    "all" => RelationalFunction::All,
                    "any" => RelationalFunction::Any,
                    _ => unreachable!(),
                };

                Ok(Some(
                    self.parse_relational_fun(ctx, body, name, &args, fun, meta)?,
                ))
            }
            _ => {
                let declarations = self.lookup_function.get(&name).ok_or_else(|| {
                    ErrorKind::SemanticError(meta, format!("Unknown function '{}'", name).into())
                })?;

                let mut maybe_decl = None;
                let mut ambiguous = false;

                'outer: for decl in declarations {
                    if args.len() != decl.parameters.len() {
                        continue;
                    }

                    let mut exact = true;

                    for ((i, decl_arg), call_arg) in
                        decl.parameters.iter().enumerate().zip(args.iter())
                    {
                        if decl.parameters_info[i].depth {
                            sampled_to_depth(&mut self.module, ctx, *call_arg)?;
                            self.invalidate_expression(ctx, call_arg.0, call_arg.1)?
                        }

                        let decl_inner = &self.module.types[*decl_arg].inner;
                        let call_inner = self.resolve_type(ctx, call_arg.0, call_arg.1)?;

                        if decl_inner == call_inner {
                            continue;
                        }

                        exact = false;

                        let (decl_kind, call_kind) = match (decl_inner, call_inner) {
                            (
                                &TypeInner::Scalar {
                                    kind: decl_kind, ..
                                },
                                &TypeInner::Scalar {
                                    kind: call_kind, ..
                                },
                            ) => (decl_kind, call_kind),
                            (
                                &TypeInner::Vector {
                                    kind: decl_kind,
                                    size: decl_size,
                                    ..
                                },
                                &TypeInner::Vector {
                                    kind: call_kind,
                                    size: call_size,
                                    ..
                                },
                            ) if decl_size == call_size => (decl_kind, call_kind),
                            (
                                &TypeInner::Matrix {
                                    rows: decl_rows,
                                    columns: decl_columns,
                                    ..
                                },
                                &TypeInner::Matrix {
                                    rows: call_rows,
                                    columns: call_columns,
                                    ..
                                },
                            ) if decl_columns == call_columns && decl_rows == call_rows => {
                                (ScalarKind::Float, ScalarKind::Float)
                            }
                            _ => continue 'outer,
                        };

                        match (type_power(decl_kind), type_power(call_kind)) {
                            (Some(decl_power), Some(call_power)) if decl_power > call_power => {}
                            _ => continue 'outer,
                        }
                    }

                    if exact {
                        maybe_decl = Some(decl);
                        ambiguous = false;
                        break;
                    } else if maybe_decl.is_some() {
                        ambiguous = true;
                    } else {
                        maybe_decl = Some(decl)
                    }
                }

                if ambiguous {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        format!("Ambiguous best function for '{}'", name).into(),
                    ));
                }

                let decl = maybe_decl.ok_or_else(|| {
                    ErrorKind::SemanticError(meta, format!("Unknown function '{}'", name).into())
                })?;

                let parameters_info = decl.parameters_info.clone();
                let parameters = decl.parameters.clone();
                let function = decl.handle;
                let is_void = decl.void;

                let mut arguments = Vec::with_capacity(args.len());
                let mut proxy_writes = Vec::new();
                for (parameter_info, (expr, parameter)) in parameters_info
                    .iter()
                    .zip(raw_args.iter().zip(parameters.iter()))
                {
                    let (mut handle, meta) =
                        ctx.lower_expect(self, *expr, parameter_info.qualifier.is_lhs(), body)?;

                    if let TypeInner::Vector { size, kind, width } =
                        *self.resolve_type(ctx, handle, meta)?
                    {
                        if parameter_info.qualifier.is_lhs()
                            && matches!(*ctx.get_expression(handle), Expression::Swizzle { .. })
                        {
                            let ty = self.module.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Vector { size, kind, width },
                            });
                            let temp_var = ctx.locals.append(LocalVariable {
                                name: None,
                                ty,
                                init: None,
                            });
                            let temp_expr =
                                ctx.add_expression(Expression::LocalVariable(temp_var), body);

                            body.push(Statement::Store {
                                pointer: temp_expr,
                                value: handle,
                            });

                            arguments.push(temp_expr);
                            proxy_writes.push((*expr, temp_expr));
                            continue;
                        }
                    }

                    let scalar_components = scalar_components(&self.module.types[*parameter].inner);
                    if let Some((kind, width)) = scalar_components {
                        ctx.implicit_conversion(self, &mut handle, meta, kind, width)?;
                    }

                    arguments.push(handle)
                }

                ctx.emit_flush(body);

                let result = if !is_void {
                    Some(ctx.add_expression(Expression::Call(function), body))
                } else {
                    None
                };

                body.push(crate::Statement::Call {
                    function,
                    arguments,
                    result,
                });

                ctx.emit_start();
                for (tgt, pointer) in proxy_writes {
                    let temp_ref = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::Variable(VariableReference {
                            expr: pointer,
                            load: true,
                            mutable: true,
                            entry_arg: None,
                        }),
                        meta,
                    });
                    let assign = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::Assign {
                            tgt,
                            value: temp_ref,
                        },
                        meta,
                    });

                    let _ = ctx.lower_expect(self, assign, false, body)?;
                }
                ctx.emit_flush(body);
                ctx.emit_start();

                Ok(result)
            }
        }
    }

    pub fn parse_relational_fun(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        name: String,
        args: &[(Handle<Expression>, SourceMetadata)],
        fun: RelationalFunction,
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        if args.len() != 1 {
            return Err(ErrorKind::wrong_function_args(name, 1, args.len(), meta));
        }

        Ok(ctx.add_expression(
            Expression::Relational {
                fun,
                argument: args[0].0,
            },
            body,
        ))
    }

    pub fn add_function(
        &mut self,
        mut function: Function,
        name: String,
        // Normalized function parameters, modifiers are not applied
        parameters: Vec<Handle<Type>>,
        parameters_info: Vec<ParameterInfo>,
        meta: SourceMetadata,
    ) -> Result<(), ErrorKind> {
        ensure_block_returns(&mut function.body);

        if name.as_str() == "main" {
            let handle = self.module.functions.append(function);
            return if self.entry_point.replace(handle).is_some() {
                Err(ErrorKind::SemanticError(meta, "main defined twice".into()))
            } else {
                Ok(())
            };
        }

        let void = function.result.is_none();

        let &mut Program {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        let declarations = lookup_function.entry(name).or_default();

        'outer: for decl in declarations.iter_mut() {
            if parameters.len() != decl.parameters.len() {
                continue;
            }

            for (new_parameter, old_parameter) in parameters.iter().zip(decl.parameters.iter()) {
                let new_inner = &module.types[*new_parameter].inner;
                let old_inner = &module.types[*old_parameter].inner;

                if new_inner != old_inner {
                    continue 'outer;
                }
            }

            if decl.defined {
                return Err(ErrorKind::SemanticError(
                    meta,
                    "Function already defined".into(),
                ));
            }

            decl.defined = true;
            decl.parameters_info = parameters_info;
            *self.module.functions.get_mut(decl.handle) = function;
            return Ok(());
        }

        let handle = module.functions.append(function);
        declarations.push(FunctionDeclaration {
            parameters,
            parameters_info,
            handle,
            defined: true,
            void,
        });

        Ok(())
    }

    pub fn add_prototype(
        &mut self,
        function: Function,
        name: String,
        // Normalized function parameters, modifiers are not applied
        parameters: Vec<Handle<Type>>,
        parameters_info: Vec<ParameterInfo>,
        meta: SourceMetadata,
    ) -> Result<(), ErrorKind> {
        let void = function.result.is_none();

        let &mut Program {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        let declarations = lookup_function.entry(name).or_default();

        'outer: for decl in declarations.iter_mut() {
            if parameters.len() != decl.parameters.len() {
                continue;
            }

            for (new_parameter, old_parameter) in parameters.iter().zip(decl.parameters.iter()) {
                let new_inner = &module.types[*new_parameter].inner;
                let old_inner = &module.types[*old_parameter].inner;

                if new_inner != old_inner {
                    continue 'outer;
                }
            }

            return Err(ErrorKind::SemanticError(
                meta,
                "Prototype already defined".into(),
            ));
        }

        let handle = module.functions.append(function);
        declarations.push(FunctionDeclaration {
            parameters,
            parameters_info,
            handle,
            defined: false,
            void,
        });

        Ok(())
    }

    pub fn add_entry_point(
        &mut self,
        function: Handle<Function>,
        mut global_init_body: Block,
        mut expressions: Arena<Expression>,
    ) -> Result<(), ErrorKind> {
        let mut arguments = Vec::new();
        let mut body = Block::with_capacity(
            // global init body
            global_init_body.len() +
                        // prologue and epilogue
                        self.entry_args.len() * 2
                        // Call, Emit for composing struct and return
                        + 3,
        );

        for arg in self.entry_args.iter() {
            if arg.storage != StorageQualifier::Input {
                continue;
            }

            let ty = self.module.global_variables[arg.handle].ty;
            let idx = arguments.len() as u32;

            arguments.push(FunctionArgument {
                name: arg.name.clone(),
                ty,
                binding: Some(arg.binding.clone()),
            });

            let pointer = expressions.append(Expression::GlobalVariable(arg.handle));
            let value = expressions.append(Expression::FunctionArgument(idx));

            body.push(Statement::Store { pointer, value });
        }

        body.append(&mut global_init_body);

        body.push(Statement::Call {
            function,
            arguments: Vec::new(),
            result: None,
        });

        let mut span = 0;
        let mut members = Vec::new();
        let mut components = Vec::new();

        for arg in self.entry_args.iter() {
            if arg.storage != StorageQualifier::Output {
                continue;
            }

            let ty = self.module.global_variables[arg.handle].ty;

            members.push(StructMember {
                name: arg.name.clone(),
                ty,
                binding: Some(arg.binding.clone()),
                offset: span,
            });

            span += self.module.types[ty].inner.span(&self.module.constants);

            let pointer = expressions.append(Expression::GlobalVariable(arg.handle));
            let len = expressions.len();
            let load = expressions.append(Expression::Load { pointer });
            body.push(Statement::Emit(expressions.range_from(len)));
            components.push(load)
        }

        let (ty, value) = if !components.is_empty() {
            let ty = self.module.types.append(Type {
                name: None,
                inner: TypeInner::Struct {
                    top_level: false,
                    members,
                    span,
                },
            });

            let len = expressions.len();
            let res = expressions.append(Expression::Compose { ty, components });
            body.push(Statement::Emit(expressions.range_from(len)));

            (Some(ty), Some(res))
        } else {
            (None, None)
        };

        body.push(Statement::Return { value });

        self.module.entry_points.push(EntryPoint {
            name: "main".to_string(),
            stage: self.stage,
            early_depth_test: Some(crate::EarlyDepthTest { conservative: None })
                .filter(|_| self.early_fragment_tests),
            workgroup_size: self.workgroup_size,
            function: Function {
                arguments,
                expressions,
                body,
                result: ty.map(|ty| FunctionResult { ty, binding: None }),
                ..Default::default()
            },
        });

        Ok(())
    }

    /// Helper function for texture calls, splits the vector argument into it's components
    fn coordinate_components(
        &self,
        ctx: &mut Context,
        (image, image_meta): (Handle<Expression>, SourceMetadata),
        (coord, coord_meta): (Handle<Expression>, SourceMetadata),
        body: &mut Block,
    ) -> Result<CoordComponents, ErrorKind> {
        if let TypeInner::Image {
            dim,
            arrayed,
            class,
        } = *self.resolve_type(ctx, image, image_meta)?
        {
            let image_size = match dim {
                ImageDimension::D1 => None,
                ImageDimension::D2 => Some(VectorSize::Bi),
                ImageDimension::D3 => Some(VectorSize::Tri),
                ImageDimension::Cube => Some(VectorSize::Tri),
            };
            let coord_size = match *self.resolve_type(ctx, coord, coord_meta)? {
                TypeInner::Vector { size, .. } => Some(size),
                _ => None,
            };
            let shadow = match class {
                ImageClass::Depth { .. } => true,
                _ => false,
            };

            let coordinate = match (image_size, coord_size) {
                (Some(size), Some(coord_s)) if size != coord_s => {
                    ctx.vector_resize(size, coord, body)
                }
                (None, Some(_)) => ctx.add_expression(
                    Expression::AccessIndex {
                        base: coord,
                        index: 0,
                    },
                    body,
                ),
                _ => coord,
            };
            let array_index = match arrayed {
                true => {
                    let index = match shadow {
                        true => image_size.map_or(0, |s| s as u32 - 1),
                        false => image_size.map_or(0, |s| s as u32),
                    };

                    Some(ctx.add_expression(Expression::AccessIndex { base: coord, index }, body))
                }
                _ => None,
            };
            let depth_ref = match shadow {
                true => {
                    let index = image_size.map_or(0, |s| s as u32);

                    Some(ctx.add_expression(Expression::AccessIndex { base: coord, index }, body))
                }
                false => None,
            };

            Ok(CoordComponents {
                coordinate,
                depth_ref,
                array_index,
            })
        } else {
            Err(ErrorKind::SemanticError(
                image_meta,
                "Type is not an image".into(),
            ))
        }
    }
}

/// Helper function to cast a expression holding a sampled image to a
/// depth image.
fn sampled_to_depth(
    module: &mut Module,
    ctx: &mut Context,
    (image, meta): (Handle<Expression>, SourceMetadata),
) -> Result<(), ErrorKind> {
    let ty = match ctx[image] {
        Expression::GlobalVariable(handle) => &mut module.global_variables.get_mut(handle).ty,
        Expression::FunctionArgument(i) => {
            ctx.parameters_info[i as usize].depth = true;
            &mut ctx.arguments[i as usize].ty
        }
        _ => {
            return Err(ErrorKind::SemanticError(
                meta,
                "Not a valid texture expression".into(),
            ))
        }
    };
    match module.types[*ty].inner {
        TypeInner::Image {
            class,
            dim,
            arrayed,
        } => match class {
            ImageClass::Sampled { multi, .. } => {
                *ty = module.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Image {
                        dim,
                        arrayed,
                        class: ImageClass::Depth { multi },
                    },
                })
            }
            ImageClass::Depth { .. } => {}
            _ => return Err(ErrorKind::SemanticError(meta, "Not a texture".into())),
        },
        _ => return Err(ErrorKind::SemanticError(meta, "Not a texture".into())),
    };

    Ok(())
}
