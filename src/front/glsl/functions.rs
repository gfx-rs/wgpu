use crate::{
    proc::ensure_block_returns, Arena, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, FunctionResult, Handle, MathFunction,
    RelationalFunction, SampleLevel, ScalarKind, ShaderStage, Statement, StorageClass,
    StructMember, Type, TypeInner,
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
    ) -> Result<Handle<Expression>, ErrorKind> {
        let args: Vec<_> = raw_args
            .iter()
            .map(|e| ctx.lower(self, *e, false, body))
            .collect::<Result<_, _>>()?;

        match fc {
            FunctionCallKind::TypeConstructor(ty) => {
                let h = if args.len() == 1 {
                    let is_vec = match *self.resolve_type(ctx, args[0].0, args[0].1)? {
                        TypeInner::Vector { .. } => true,
                        _ => false,
                    };

                    match self.module.types[ty].inner {
                        TypeInner::Vector { size, .. } if !is_vec => {
                            ctx.expressions.append(Expression::Splat {
                                size,
                                value: args[0].0,
                            })
                        }
                        TypeInner::Scalar { kind, width }
                        | TypeInner::Vector { kind, width, .. } => {
                            ctx.expressions.append(Expression::As {
                                kind,
                                expr: args[0].0,
                                convert: Some(width),
                            })
                        }
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            let value = ctx.expressions.append(Expression::As {
                                kind: ScalarKind::Float,
                                expr: args[0].0,
                                convert: Some(width),
                            });

                            let column = if is_vec {
                                value
                            } else {
                                ctx.expressions
                                    .append(Expression::Splat { size: rows, value })
                            };

                            let columns =
                                std::iter::repeat(column).take(columns as usize).collect();

                            ctx.expressions.append(Expression::Compose {
                                ty,
                                components: columns,
                            })
                        }
                        _ => return Err(ErrorKind::SemanticError(meta, "Bad cast".into())),
                    }
                } else {
                    ctx.expressions.append(Expression::Compose {
                        ty,
                        components: args.iter().map(|e| e.0).collect(),
                    })
                };

                Ok(h)
            }
            FunctionCallKind::Function(name) => {
                match name.as_str() {
                    "sampler2D" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        ctx.samplers.insert(args[0].0, args[1].0);
                        Ok(args[0].0)
                    }
                    "texture" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                            Ok(ctx.expressions.append(Expression::ImageSample {
                                image: args[0].0,
                                sampler,
                                coordinate: args[1].0,
                                array_index: None, //TODO
                                offset: None,      //TODO
                                level: SampleLevel::Auto,
                                depth_ref: None,
                            }))
                        } else {
                            Err(ErrorKind::SemanticError(meta, "Bad call to texture".into()))
                        }
                    }
                    "textureLod" => {
                        if args.len() != 3 {
                            return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                        }
                        if let Some(sampler) = ctx.samplers.get(&args[0].0).copied() {
                            Ok(ctx.expressions.append(Expression::ImageSample {
                                image: args[0].0,
                                sampler,
                                coordinate: args[1].0,
                                array_index: None, //TODO
                                offset: None,      //TODO
                                level: SampleLevel::Exact(args[2].0),
                                depth_ref: None,
                            }))
                        } else {
                            Err(ErrorKind::SemanticError(
                                meta,
                                "Bad call to textureLod".into(),
                            ))
                        }
                    }
                    "ceil" | "round" | "floor" | "fract" | "trunc" | "sin" | "abs" | "sqrt"
                    | "inversesqrt" | "exp" | "exp2" | "sign" | "transpose" | "inverse"
                    | "normalize" => {
                        if args.len() != 1 {
                            return Err(ErrorKind::wrong_function_args(name, 1, args.len(), meta));
                        }
                        Ok(ctx.expressions.append(Expression::Math {
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
                                _ => unreachable!(),
                            },
                            arg: args[0].0,
                            arg1: None,
                            arg2: None,
                        }))
                    }
                    "pow" | "dot" | "max" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        Ok(ctx.expressions.append(Expression::Math {
                            fun: match name.as_str() {
                                "pow" => MathFunction::Pow,
                                "dot" => MathFunction::Dot,
                                "max" => MathFunction::Max,
                                _ => unreachable!(),
                            },
                            arg: args[0].0,
                            arg1: Some(args[1].0),
                            arg2: None,
                        }))
                    }
                    "mix" | "clamp" => {
                        if args.len() != 3 {
                            return Err(ErrorKind::wrong_function_args(name, 3, args.len(), meta));
                        }
                        Ok(ctx.expressions.append(Expression::Math {
                            fun: match name.as_str() {
                                "mix" => MathFunction::Mix,
                                "clamp" => MathFunction::Clamp,
                                _ => unreachable!(),
                            },
                            arg: args[0].0,
                            arg1: Some(args[1].0),
                            arg2: Some(args[2].0),
                        }))
                    }
                    "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
                    | "notEqual" => {
                        if args.len() != 2 {
                            return Err(ErrorKind::wrong_function_args(name, 2, args.len(), meta));
                        }
                        Ok(ctx.expressions.append(Expression::Binary {
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
                        }))
                    }
                    "isinf" => {
                        self.parse_relational_fun(ctx, name, &args, RelationalFunction::IsInf, meta)
                    }
                    "isnan" => {
                        self.parse_relational_fun(ctx, name, &args, RelationalFunction::IsNan, meta)
                    }
                    "all" => {
                        self.parse_relational_fun(ctx, name, &args, RelationalFunction::All, meta)
                    }
                    "any" => {
                        self.parse_relational_fun(ctx, name, &args, RelationalFunction::Any, meta)
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
                        for (qualifier, expr) in fun.parameters.iter().zip(raw_args.iter()) {
                            let handle = ctx.lower(self, *expr, qualifier.is_lhs(), body)?.0;
                            arguments.push(handle)
                        }

                        let expression = ctx.expressions.append(Expression::Call(fun.handle));
                        body.push(crate::Statement::Call {
                            function: fun.handle,
                            arguments,
                            result: Some(expression),
                        });
                        Ok(expression)
                    }
                }
            }
        }
    }

    pub fn parse_relational_fun(
        &mut self,
        ctx: &mut Context,
        name: String,
        args: &[(Handle<Expression>, SourceMetadata)],
        fun: RelationalFunction,
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        if args.len() != 1 {
            return Err(ErrorKind::wrong_function_args(name, 1, args.len(), meta));
        }

        Ok(ctx.expressions.append(Expression::Relational {
            fun,
            argument: args[0].0,
        }))
    }

    pub fn add_function(
        &mut self,
        mut function: Function,
        parameters: Vec<ParameterQualifier>,
        meta: SourceMetadata,
    ) -> Result<(), ErrorKind> {
        ensure_block_returns(&mut function.body);
        let name = function
            .name
            .clone()
            .ok_or_else(|| ErrorKind::SemanticError(meta, "Unnamed function".into()))?;
        let stage = self.entry_points.get(&name);

        if let Some(&stage) = stage {
            let handle = self.module.functions.append(function);
            self.entries.push((name, stage, handle));
        } else {
            let sig = FunctionSignature {
                name,
                parameters: function.arguments.iter().map(|p| p.ty).collect(),
            };

            for (arg, qualifier) in function.arguments.iter_mut().zip(parameters.iter()) {
                if qualifier.is_lhs() {
                    arg.ty = self.module.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Pointer {
                            base: arg.ty,
                            class: StorageClass::Function,
                        },
                    })
                }
            }

            if let Some(decl) = self.lookup_function.get_mut(&sig) {
                if decl.defined {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Function already defined".into(),
                    ));
                }

                decl.defined = true;
                *self.module.functions.get_mut(decl.handle) = function;
            } else {
                let handle = self.module.functions.append(function);
                self.lookup_function.insert(
                    sig,
                    FunctionDeclaration {
                        parameters,
                        handle,
                        defined: true,
                    },
                );
            }
        }

        Ok(())
    }

    pub fn add_prototype(
        &mut self,
        mut function: Function,
        parameters: Vec<ParameterQualifier>,
        meta: SourceMetadata,
    ) -> Result<(), ErrorKind> {
        let name = function
            .name
            .clone()
            .ok_or_else(|| ErrorKind::SemanticError(meta, "Unnamed function".into()))?;
        let sig = FunctionSignature {
            name,
            parameters: function.arguments.iter().map(|p| p.ty).collect(),
        };

        for (arg, qualifier) in function.arguments.iter_mut().zip(parameters.iter()) {
            if qualifier.is_lhs() {
                arg.ty = self.module.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Pointer {
                        base: arg.ty,
                        class: StorageClass::Function,
                    },
                })
            }
        }

        let handle = self.module.functions.append(function);

        self.lookup_function.insert(
            sig,
            FunctionDeclaration {
                parameters,
                handle,
                defined: false,
            },
        );

        Ok(())
    }

    pub fn add_entry_points(&mut self) {
        for (name, stage, function) in self.entries.iter().cloned() {
            let mut arguments = Vec::new();
            let mut expressions = Arena::new();
            let mut body = Vec::new();

            for (binding, input, handle) in self.entry_args.iter().cloned() {
                match binding {
                    Binding::Location { .. } if !input => continue,
                    _ => {}
                }

                let ty = self.module.global_variables[handle].ty;
                let arg = arguments.len() as u32;

                arguments.push(FunctionArgument {
                    name: None,
                    ty,
                    binding: Some(binding),
                });

                let pointer = expressions.append(Expression::GlobalVariable(handle));
                let value = expressions.append(Expression::FunctionArgument(arg));

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

            for (binding, input, handle) in self.entry_args.iter().cloned() {
                match binding {
                    Binding::Location { .. } if input => continue,
                    Binding::BuiltIn(builtin) if !should_write(builtin, stage) => continue,
                    _ => {}
                }

                let ty = self.module.global_variables[handle].ty;

                members.push(StructMember {
                    name: None,
                    ty,
                    binding: Some(binding),
                    offset: span,
                });

                span += self.module.types[ty].inner.span(&self.module.constants);

                let pointer = expressions.append(Expression::GlobalVariable(handle));
                let load = expressions.append(Expression::Load { pointer });
                components.push(load)
            }

            let ty = self.module.types.append(Type {
                name: None,
                inner: TypeInner::Struct {
                    top_level: true,
                    members,
                    span,
                },
            });

            let res = expressions.append(Expression::Compose { ty, components });
            body.push(Statement::Return { value: Some(res) });

            self.module.entry_points.push(EntryPoint {
                name,
                stage,
                // TODO
                early_depth_test: None,
                workgroup_size: [0; 3],
                function: Function {
                    arguments,
                    expressions,
                    body,
                    result: Some(FunctionResult { ty, binding: None }),
                    ..Default::default()
                },
            });
        }
    }
}

fn should_write(built_in: BuiltIn, stage: ShaderStage) -> bool {
    match (built_in, stage) {
        (BuiltIn::Position, ShaderStage::Vertex)
        | (BuiltIn::ClipDistance, ShaderStage::Vertex)
        | (BuiltIn::PointSize, ShaderStage::Vertex)
        | (BuiltIn::FragDepth, ShaderStage::Fragment)
        | (BuiltIn::SampleMask, ShaderStage::Fragment) => true,
        _ => false,
    }
}
