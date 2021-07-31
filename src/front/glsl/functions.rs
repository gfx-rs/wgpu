use bit_set::BitSet;

use crate::{
    proc::ensure_block_returns, Arena, BinaryOperator, Block, Constant, ConstantInner, EntryPoint,
    Expression, Function, FunctionArgument, FunctionResult, Handle, ImageQuery, LocalVariable,
    MathFunction, RelationalFunction, SampleLevel, ScalarKind, ScalarValue, ShaderStage, Statement,
    StructMember, SwizzleComponent, Type, TypeInner, VectorSize,
};

use super::{ast::*, error::ErrorKind, SourceMetadata};

impl Program<'_> {
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
                            let expr = ctx.add_expression(
                                Expression::Swizzle {
                                    size,
                                    vector: args[0].0,
                                    pattern: SwizzleComponent::XYZW,
                                },
                                body,
                            );

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
                                TypeInner::Matrix { .. } => {
                                    let mut components = Vec::new();

                                    for n in 0..columns as u32 {
                                        let vector = ctx.add_expression(
                                            Expression::AccessIndex {
                                                base: value,
                                                index: n,
                                            },
                                            body,
                                        );

                                        let c = ctx.add_expression(
                                            Expression::Swizzle {
                                                size: rows,
                                                vector,
                                                pattern: SwizzleComponent::XYZW,
                                            },
                                            body,
                                        );

                                        components.push(c)
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

                let ty = match ctx[args[0].0] {
                    crate::Expression::GlobalVariable(handle) => {
                        &mut self.module.global_variables.get_mut(handle).ty
                    }
                    crate::Expression::FunctionArgument(i) => {
                        ctx.depth_set.insert(i as usize);
                        &mut ctx.arguments[i as usize].ty
                    }
                    _ => {
                        return Err(ErrorKind::SemanticError(
                            args[0].1,
                            "Not a valid texture expression".into(),
                        ))
                    }
                };
                match self.module.types[*ty].inner {
                    TypeInner::Image {
                        class,
                        dim,
                        arrayed,
                    } => match class {
                        crate::ImageClass::Sampled { multi, .. } => {
                            *ty = self.module.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Image {
                                    dim,
                                    arrayed,
                                    class: crate::ImageClass::Depth { multi },
                                },
                            })
                        }
                        crate::ImageClass::Depth { .. } => {}
                        _ => {
                            return Err(ErrorKind::SemanticError(args[0].1, "Not a texture".into()))
                        }
                    },
                    _ => return Err(ErrorKind::SemanticError(args[0].1, "Not a texture".into())),
                };

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
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(
                        ctx.add_expression(
                            Expression::ImageSample {
                                image: args[0].0,
                                sampler,
                                coordinate: args[1].0,
                                array_index: None, //TODO
                                offset: None,
                                level: args.get(2).map_or(SampleLevel::Auto, |&(expr, _)| {
                                    SampleLevel::Bias(expr)
                                }),
                                depth_ref: None,
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
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: args[1].0,
                            array_index: None, //TODO
                            offset: None,
                            level: SampleLevel::Exact(args[2].0),
                            depth_ref: None,
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
                let dim = match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                    TypeInner::Image { dim, .. } => match dim {
                        crate::ImageDimension::D1 => 1,
                        crate::ImageDimension::D2 => 2,
                        crate::ImageDimension::D3 => 3,
                        crate::ImageDimension::Cube => {
                            return Err(ErrorKind::SemanticError(
                                meta,
                                "textureProj doesn't accept cube texture".into(),
                            ))
                        }
                    },
                    _ => {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Bad call to textureProj".into(),
                        ))
                    }
                };
                match *self.resolve_type(ctx, args[1].0, args[1].1)? {
                    TypeInner::Vector { size, .. } => {
                        if !(size as usize + 1 == dim || size == VectorSize::Quad) {
                            return Err(ErrorKind::SemanticError(
                                meta,
                                "Bad call to textureProj".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Bad call to textureProj".into(),
                        ))
                    }
                }
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: args[1].0,
                            array_index: None, //TODO
                            offset: None,
                            level,
                            depth_ref: None,
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
                if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                    Ok(Some(ctx.add_expression(
                        Expression::ImageSample {
                            image: args[0].0,
                            sampler,
                            coordinate: args[1].0,
                            array_index: None, //TODO
                            offset: None,
                            level: SampleLevel::Gradient {
                                x: args[2].0,
                                y: args[3].0,
                            },
                            depth_ref: None,
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
                if ctx.samplers.get(&args[0].0).is_some() {
                    let (arrayed, dims) = match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                        TypeInner::Image { arrayed, dim, .. } => (arrayed, dim),
                        _ => (false, crate::ImageDimension::D1),
                    };

                    let (coordinate, array_index) = if arrayed {
                        (
                            match dims {
                                crate::ImageDimension::D1 => ctx.add_expression(
                                    Expression::AccessIndex {
                                        base: args[1].0,
                                        index: 0,
                                    },
                                    body,
                                ),
                                crate::ImageDimension::D2 => ctx.add_expression(
                                    Expression::Swizzle {
                                        size: VectorSize::Bi,
                                        vector: args[1].0,
                                        pattern: SwizzleComponent::XYZW,
                                    },
                                    body,
                                ),
                                _ => ctx.add_expression(
                                    Expression::Swizzle {
                                        size: VectorSize::Tri,
                                        vector: args[1].0,
                                        pattern: SwizzleComponent::XYZW,
                                    },
                                    body,
                                ),
                            },
                            Some(ctx.add_expression(
                                Expression::AccessIndex {
                                    base: args[1].0,
                                    index: match dims {
                                        crate::ImageDimension::D1 => 1,
                                        crate::ImageDimension::D2 => 2,
                                        crate::ImageDimension::D3 => 3,
                                        crate::ImageDimension::Cube => 2,
                                    },
                                },
                                body,
                            )),
                        )
                    } else {
                        (args[1].0, None)
                    };

                    Ok(Some(ctx.add_expression(
                        Expression::ImageLoad {
                            image: args[0].0,
                            coordinate,
                            array_index,
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
            "max" => {
                if args.len() != 2 {
                    return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                }

                let (mut arg0, arg0_meta) = args[0];
                let (mut arg1, arg1_meta) = args[1];

                let arg0_size = match *self.resolve_type(ctx, arg0, arg0_meta)? {
                    TypeInner::Vector { size, .. } => Some(size),
                    _ => None,
                };
                let arg1_size = match *self.resolve_type(ctx, arg1, arg1_meta)? {
                    TypeInner::Vector { size, .. } => Some(size),
                    _ => None,
                };

                ctx.binary_implicit_conversion(self, &mut arg0, arg0_meta, &mut arg1, arg1_meta)?;

                ctx.implicit_splat(self, &mut arg0, arg0_meta, arg1_size)?;
                ctx.implicit_splat(self, &mut arg1, arg1_meta, arg0_size)?;

                Ok(Some(ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::Max,
                        arg: arg0,
                        arg1: Some(arg1),
                        arg2: None,
                    },
                    body,
                )))
            }
            "pow" | "dot" | "min" | "reflect" | "cross" | "outerProduct" | "distance" | "step"
            | "modf" | "frexp" | "ldexp" => {
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
                            "min" => MathFunction::Min,
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
                        if decl.depth_set.contains(i) {
                            let ty = match ctx[args[0].0] {
                                crate::Expression::GlobalVariable(handle) => {
                                    &mut self.module.global_variables.get_mut(handle).ty
                                }
                                crate::Expression::FunctionArgument(i) => {
                                    ctx.depth_set.insert(i as usize);
                                    &mut ctx.arguments[i as usize].ty
                                }
                                _ => {
                                    return Err(ErrorKind::SemanticError(
                                        args[0].1,
                                        "Not a valid texture expression".into(),
                                    ))
                                }
                            };
                            match self.module.types[*ty].inner {
                                TypeInner::Image {
                                    class,
                                    dim,
                                    arrayed,
                                } => match class {
                                    crate::ImageClass::Sampled { multi, .. } => {
                                        *ty = self.module.types.fetch_or_append(Type {
                                            name: None,
                                            inner: TypeInner::Image {
                                                dim,
                                                arrayed,
                                                class: crate::ImageClass::Depth { multi },
                                            },
                                        })
                                    }
                                    crate::ImageClass::Depth { .. } => {}
                                    _ => {
                                        return Err(ErrorKind::SemanticError(
                                            args[0].1,
                                            "Not a texture".into(),
                                        ))
                                    }
                                },
                                _ => {
                                    return Err(ErrorKind::SemanticError(
                                        args[0].1,
                                        "Not a texture".into(),
                                    ))
                                }
                            };
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

                let qualifiers = decl.qualifiers.clone();
                let parameters = decl.parameters.clone();
                let function = decl.handle;
                let is_void = decl.void;

                let mut arguments = Vec::with_capacity(args.len());
                let mut proxy_writes = Vec::new();
                for (qualifier, (expr, parameter)) in qualifiers
                    .iter()
                    .zip(raw_args.iter().zip(parameters.iter()))
                {
                    let (mut handle, meta) =
                        ctx.lower_expect(self, *expr, qualifier.is_lhs(), body)?;

                    if let TypeInner::Vector { size, kind, width } =
                        *self.resolve_type(ctx, handle, meta)?
                    {
                        if qualifier.is_lhs()
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
        depth_set: BitSet,
        qualifiers: Vec<ParameterQualifier>,
        meta: SourceMetadata,
    ) -> Result<Handle<Function>, ErrorKind> {
        ensure_block_returns(&mut function.body);
        let stage = self.entry_points.get(&name);

        Ok(if let Some(&stage) = stage {
            let handle = self.module.functions.append(function);
            self.entries.push((name, stage, handle));
            self.function_arg_use.push(Vec::new());
            handle
        } else {
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

                for (new_parameter, old_parameter) in parameters.iter().zip(decl.parameters.iter())
                {
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
                decl.qualifiers = qualifiers;
                decl.depth_set = depth_set;
                *self.module.functions.get_mut(decl.handle) = function;
                return Ok(decl.handle);
            }

            self.function_arg_use.push(Vec::new());
            let handle = module.functions.append(function);
            declarations.push(FunctionDeclaration {
                parameters,
                qualifiers,
                handle,
                defined: true,
                void,
                depth_set,
            });
            handle
        })
    }

    pub fn add_prototype(
        &mut self,
        function: Function,
        name: String,
        // Normalized function parameters, modifiers are not applied
        parameters: Vec<Handle<Type>>,
        qualifiers: Vec<ParameterQualifier>,
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

        self.function_arg_use.push(Vec::new());
        let handle = module.functions.append(function);
        declarations.push(FunctionDeclaration {
            parameters,
            qualifiers,
            handle,
            defined: false,
            void,
            depth_set: BitSet::new(),
        });

        Ok(())
    }

    fn check_call_global(
        &self,
        caller: Handle<Function>,
        function_arg_use: &mut [Vec<EntryArgUse>],
        stmt: &Statement,
    ) {
        match *stmt {
            Statement::Block(ref block) => {
                for stmt in block {
                    self.check_call_global(caller, function_arg_use, stmt)
                }
            }
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => {
                for stmt in accept.iter().chain(reject.iter()) {
                    self.check_call_global(caller, function_arg_use, stmt)
                }
            }
            Statement::Switch {
                ref cases,
                ref default,
                ..
            } => {
                for stmt in cases
                    .iter()
                    .flat_map(|c| c.body.iter())
                    .chain(default.iter())
                {
                    self.check_call_global(caller, function_arg_use, stmt)
                }
            }
            Statement::Loop {
                ref body,
                ref continuing,
            } => {
                for stmt in body.iter().chain(continuing.iter()) {
                    self.check_call_global(caller, function_arg_use, stmt)
                }
            }
            Statement::Call { function, .. } => {
                let callee_len = function_arg_use[function.index()].len();
                let caller_len = function_arg_use[caller.index()].len();
                function_arg_use[caller.index()].extend(
                    std::iter::repeat(EntryArgUse::empty())
                        .take(callee_len.saturating_sub(caller_len)),
                );

                for i in 0..callee_len.min(caller_len) {
                    let callee_use = function_arg_use[function.index()][i];
                    function_arg_use[caller.index()][i] |= callee_use
                }
            }
            _ => {}
        }
    }

    pub fn add_entry_points(&mut self) {
        let mut function_arg_use = Vec::new();
        std::mem::swap(&mut self.function_arg_use, &mut function_arg_use);

        for (handle, function) in self.module.functions.iter() {
            for stmt in function.body.iter() {
                self.check_call_global(handle, &mut function_arg_use, stmt)
            }
        }

        for (name, stage, function) in self.entries.iter().cloned() {
            let mut arguments = Vec::new();
            let mut expressions = Arena::new();
            let mut body = Vec::new();

            let can_strip_stage_inputs =
                self.strip_unused_linkages || stage != ShaderStage::Fragment;
            let can_strip_stage_outputs =
                self.strip_unused_linkages || stage != ShaderStage::Vertex;

            for (i, arg) in self.entry_args.iter().enumerate() {
                if arg.storage != StorageQualifier::Input {
                    continue;
                }

                if !arg.prologue.contains(stage.into()) {
                    continue;
                }

                let is_used = function_arg_use[function.index()]
                    .get(i)
                    .map_or(false, |u| u.contains(EntryArgUse::READ));

                if can_strip_stage_inputs && !is_used {
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

            body.push(Statement::Call {
                function,
                arguments: Vec::new(),
                result: None,
            });

            let mut span = 0;
            let mut members = Vec::new();
            let mut components = Vec::new();

            for (i, arg) in self.entry_args.iter().enumerate() {
                if arg.storage != StorageQualifier::Output {
                    continue;
                }

                let is_used = function_arg_use[function.index()]
                    .get(i)
                    .map_or(false, |u| u.contains(EntryArgUse::WRITE));

                if can_strip_stage_outputs && !is_used {
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
                name,
                stage,
                early_depth_test: Some(crate::EarlyDepthTest { conservative: None })
                    .filter(|_| self.early_fragment_tests && stage == crate::ShaderStage::Fragment),
                workgroup_size: if let crate::ShaderStage::Compute = stage {
                    self.workgroup_size
                } else {
                    [0; 3]
                },
                function: Function {
                    arguments,
                    expressions,
                    body,
                    result: ty.map(|ty| FunctionResult { ty, binding: None }),
                    ..Default::default()
                },
            });
        }
    }
}
