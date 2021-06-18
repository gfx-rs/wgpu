use crate::{
    proc::ensure_block_returns, Arena, BinaryOperator, Block, EntryPoint, Expression, Function,
    FunctionArgument, FunctionResult, Handle, LocalVariable, MathFunction, RelationalFunction,
    SampleLevel, ScalarKind, Statement, StructMember, SwizzleComponent, Type, TypeInner,
};

use super::{ast::*, error::ErrorKind, SourceMetadata};

impl Program<'_> {
    pub fn function_call(
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
                    let is_vec = match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                        TypeInner::Vector { .. } => true,
                        _ => false,
                    };

                    match self.module.types[ty].inner {
                        TypeInner::Vector { size, kind, .. } if !is_vec => {
                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(self, &mut value, meta, kind)?;

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
                        TypeInner::Matrix { columns, rows, .. } => {
                            // TODO: casts
                            // `Expression::As` doesn't support matrix width
                            // casts so we need to do some extra work for casts

                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(self, &mut value, meta, ScalarKind::Float)?;
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

                    for (mut arg, meta) in args.iter().copied() {
                        if let Some(kind) = self.module.types[ty].inner.scalar_kind() {
                            ctx.implicit_conversion(self, &mut arg, meta, kind)?;
                        }
                        components.push(arg)
                    }

                    ctx.add_expression(Expression::Compose { ty, components }, body)
                };

                Ok(Some(h))
            }
            FunctionCallKind::Function(name) => {
                match name.as_str() {
                    "sampler2D" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        ctx.samplers.insert(args[0].0, args[1].0);
                        Ok(Some(args[0].0))
                    }
                    "texture" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                            Ok(Some(ctx.add_expression(
                                Expression::ImageSample {
                                    image: args[0].0,
                                    sampler,
                                    coordinate: args[1].0,
                                    array_index: None, //TODO
                                    offset: None,      //TODO
                                    level: SampleLevel::Auto,
                                    depth_ref: None,
                                },
                                body,
                            )))
                        } else {
                            Err(ErrorKind::SemanticError(meta, "Bad call to texture".into()))
                        }
                    }
                    "textureLod" => {
                        if args.len() != 3 {
                            return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                        }
                        let exact = ctx.add_expression(
                            Expression::As {
                                kind: crate::ScalarKind::Float,
                                expr: args[2].0,
                                convert: Some(4),
                            },
                            body,
                        );
                        if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                            Ok(Some(ctx.add_expression(
                                Expression::ImageSample {
                                    image: args[0].0,
                                    sampler,
                                    coordinate: args[1].0,
                                    array_index: None, //TODO
                                    offset: None,      //TODO
                                    level: SampleLevel::Exact(exact),
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
                    "ceil" | "round" | "floor" | "fract" | "trunc" | "sin" | "abs" | "sqrt"
                    | "inversesqrt" | "exp" | "exp2" | "sign" | "transpose" | "inverse"
                    | "normalize" | "sinh" | "cos" | "cosh" | "tan" | "tanh" | "acos" | "asin"
                    | "log" | "log2" | "length" | "determinant" | "bitCount"
                    | "bitfieldReverse" => {
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
                            _ => {
                                return Err(ErrorKind::wrong_function_args(
                                    name,
                                    2,
                                    args.len(),
                                    meta,
                                ))
                            }
                        };
                        Ok(Some(ctx.add_expression(expr, body)))
                    }
                    "mod" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }

                        let (mut left, left_meta) = args[0];
                        let (mut right, right_meta) = args[1];

                        ctx.binary_implicit_conversion(
                            self, &mut left, left_meta, &mut right, right_meta,
                        )?;

                        Ok(Some(ctx.add_expression(
                            Expression::Binary {
                                op: BinaryOperator::Modulo,
                                left,
                                right,
                            },
                            body,
                        )))
                    }
                    "pow" | "dot" | "max" | "min" | "reflect" | "cross" | "outerProduct"
                    | "distance" | "step" | "modf" | "frexp" | "ldexp" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        Ok(Some(ctx.add_expression(
                            Expression::Math {
                                fun: match name.as_str() {
                                    "pow" => MathFunction::Pow,
                                    "dot" => MathFunction::Dot,
                                    "max" => MathFunction::Max,
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
                                arg: args[0].0,
                                arg1: Some(args[1].0),
                                arg2: None,
                            },
                            body,
                        )))
                    }
                    "mix" | "clamp" | "faceforward" | "refract" | "fma" | "smoothstep" => {
                        if args.len() != 3 {
                            return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                        }
                        Ok(Some(ctx.add_expression(
                            Expression::Math {
                                fun: match name.as_str() {
                                    "mix" => MathFunction::Mix,
                                    "clamp" => MathFunction::Clamp,
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
                        let mut parameters = Vec::new();

                        for (e, meta) in args {
                            let handle = self.resolve_handle(ctx, e, meta)?;

                            parameters.push(handle)
                        }

                        let sig = FunctionSignature { name, parameters };

                        let fun = self
                            .lookup_function
                            .get(&sig)
                            .ok_or_else(|| {
                                ErrorKind::SemanticError(
                                    meta,
                                    // FIXME: Proper signature display
                                    format!("Unknown function: {:?}", sig).into(),
                                )
                            })?
                            .clone();

                        let mut arguments = Vec::with_capacity(raw_args.len());
                        let mut proxy_writes = Vec::new();
                        for (qualifier, expr) in fun.qualifiers.iter().zip(raw_args.iter()) {
                            let handle = ctx.lower_expect(self, *expr, qualifier.is_lhs(), body)?.0;
                            if qualifier.is_lhs()
                                && matches! { ctx.get_expression(handle), &Expression::Swizzle { .. } }
                            {
                                let meta = ctx.hir_exprs[*expr].meta;
                                let ty = self.resolve_handle(ctx, handle, meta)?;
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
                            } else {
                                arguments.push(handle);
                            }
                        }

                        ctx.emit_flush(body);

                        let result = if !fun.void {
                            Some(ctx.add_expression(Expression::Call(fun.handle), body))
                        } else {
                            None
                        };

                        body.push(crate::Statement::Call {
                            function: fun.handle,
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
        sig: FunctionSignature,
        qualifiers: Vec<ParameterQualifier>,
        meta: SourceMetadata,
    ) -> Result<Handle<Function>, ErrorKind> {
        ensure_block_returns(&mut function.body);
        let stage = self.entry_points.get(&sig.name);

        Ok(if let Some(&stage) = stage {
            let handle = self.module.functions.append(function);
            self.entries.push((sig.name, stage, handle));
            self.function_arg_use.push(Vec::new());
            handle
        } else {
            let void = function.result.is_none();

            if let Some(decl) = self.lookup_function.get_mut(&sig) {
                if decl.defined {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Function already defined".into(),
                    ));
                }

                decl.defined = true;
                *self.module.functions.get_mut(decl.handle) = function;
                decl.handle
            } else {
                self.function_arg_use.push(Vec::new());
                let handle = self.module.functions.append(function);
                self.lookup_function.insert(
                    sig,
                    FunctionDeclaration {
                        qualifiers,
                        handle,
                        defined: true,
                        void,
                    },
                );
                handle
            }
        })
    }

    pub fn add_prototype(
        &mut self,
        function: Function,
        sig: FunctionSignature,
        qualifiers: Vec<ParameterQualifier>,
        meta: SourceMetadata,
    ) -> Result<(), ErrorKind> {
        let void = function.result.is_none();

        self.function_arg_use.push(Vec::new());
        let handle = self.module.functions.append(function);

        if self
            .lookup_function
            .insert(
                sig,
                FunctionDeclaration {
                    qualifiers,
                    handle,
                    defined: false,
                    void,
                },
            )
            .is_some()
        {
            return Err(ErrorKind::SemanticError(
                meta,
                "Prototype already defined".into(),
            ));
        }

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

            for (i, arg) in self.entry_args.iter().enumerate() {
                if function_arg_use[function.index()]
                    .get(i)
                    .map_or(true, |u| !u.contains(EntryArgUse::READ))
                    || !arg.prologue.contains(stage.into())
                {
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
                if function_arg_use[function.index()]
                    .get(i)
                    .map_or(true, |u| !u.contains(EntryArgUse::WRITE))
                {
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
