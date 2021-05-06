use super::super::Typifier;
use crate::{
    proc::ensure_block_returns, BinaryOperator, Block, EntryPoint, Expression, Function,
    MathFunction, RelationalFunction, SampleLevel, ScalarKind, TypeInner,
};

use super::{ast::*, error::ErrorKind};

impl Program<'_> {
    pub fn function_call(&mut self, fc: FunctionCall) -> Result<ExpressionRule, ErrorKind> {
        match fc.kind {
            FunctionCallKind::TypeConstructor(ty) => {
                let h = if fc.args.len() == 1 {
                    let is_vec = match *self.resolve_type(fc.args[0].expression)? {
                        TypeInner::Vector { .. } => true,
                        _ => false,
                    };

                    match self.module.types[ty].inner {
                        TypeInner::Vector { size, .. } if !is_vec => {
                            self.context.expressions.append(Expression::Splat {
                                size,
                                value: fc.args[0].expression,
                            })
                        }
                        TypeInner::Scalar { kind, width }
                        | TypeInner::Vector { kind, width, .. } => {
                            self.context.expressions.append(Expression::As {
                                kind,
                                expr: fc.args[0].expression,
                                convert: Some(width),
                            })
                        }
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            let value = self.context.expressions.append(Expression::As {
                                kind: ScalarKind::Float,
                                expr: fc.args[0].expression,
                                convert: Some(width),
                            });

                            let column = if is_vec {
                                value
                            } else {
                                self.context
                                    .expressions
                                    .append(Expression::Splat { size: rows, value })
                            };

                            let columns =
                                std::iter::repeat(column).take(columns as usize).collect();

                            self.context.expressions.append(Expression::Compose {
                                ty,
                                components: columns,
                            })
                        }
                        _ => return Err(ErrorKind::SemanticError("Bad cast".into())),
                    }
                } else {
                    self.context.expressions.append(Expression::Compose {
                        ty,
                        components: fc.args.iter().map(|a| a.expression).collect(),
                    })
                };

                Ok(ExpressionRule {
                    expression: h,
                    statements: fc
                        .args
                        .into_iter()
                        .map(|a| a.statements)
                        .flatten()
                        .collect(),
                    sampler: None,
                })
            }
            FunctionCallKind::Function(name) => {
                match name.as_str() {
                    "sampler2D" => {
                        if fc.args.len() != 2 {
                            return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
                        }
                        Ok(ExpressionRule {
                            expression: fc.args[0].expression,
                            sampler: Some(fc.args[1].expression),
                            statements: fc
                                .args
                                .into_iter()
                                .map(|a| a.statements)
                                .flatten()
                                .collect(),
                        })
                    }
                    "texture" => {
                        if fc.args.len() != 2 {
                            return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
                        }
                        if let Some(sampler) = fc.args[0].sampler {
                            Ok(ExpressionRule {
                                expression: self.context.expressions.append(
                                    Expression::ImageSample {
                                        image: fc.args[0].expression,
                                        sampler,
                                        coordinate: fc.args[1].expression,
                                        array_index: None, //TODO
                                        offset: None,      //TODO
                                        level: SampleLevel::Auto,
                                        depth_ref: None,
                                    },
                                ),
                                sampler: None,
                                statements: fc
                                    .args
                                    .into_iter()
                                    .map(|a| a.statements)
                                    .flatten()
                                    .collect(),
                            })
                        } else {
                            Err(ErrorKind::SemanticError("Bad call to texture".into()))
                        }
                    }
                    "textureLod" => {
                        if fc.args.len() != 3 {
                            return Err(ErrorKind::WrongNumberArgs(name, 3, fc.args.len()));
                        }
                        if let Some(sampler) = fc.args[0].sampler {
                            Ok(ExpressionRule {
                                expression: self.context.expressions.append(
                                    Expression::ImageSample {
                                        image: fc.args[0].expression,
                                        sampler,
                                        coordinate: fc.args[1].expression,
                                        array_index: None, //TODO
                                        offset: None,      //TODO
                                        level: SampleLevel::Exact(fc.args[2].expression),
                                        depth_ref: None,
                                    },
                                ),
                                sampler: None,
                                statements: fc
                                    .args
                                    .into_iter()
                                    .map(|a| a.statements)
                                    .flatten()
                                    .collect(),
                            })
                        } else {
                            Err(ErrorKind::SemanticError("Bad call to textureLod".into()))
                        }
                    }
                    "ceil" | "round" | "floor" | "fract" | "trunc" | "sin" | "abs" | "sqrt"
                    | "inversesqrt" | "exp" | "exp2" | "sign" | "transpose" | "inverse"
                    | "normalize" => {
                        if fc.args.len() != 1 {
                            return Err(ErrorKind::WrongNumberArgs(name, 1, fc.args.len()));
                        }
                        Ok(ExpressionRule {
                            expression: self.context.expressions.append(Expression::Math {
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
                                arg: fc.args[0].expression,
                                arg1: None,
                                arg2: None,
                            }),
                            sampler: None,
                            statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
                        })
                    }
                    "pow" | "dot" | "max" => {
                        if fc.args.len() != 2 {
                            return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
                        }
                        Ok(ExpressionRule {
                            expression: self.context.expressions.append(Expression::Math {
                                fun: match name.as_str() {
                                    "pow" => MathFunction::Pow,
                                    "dot" => MathFunction::Dot,
                                    "max" => MathFunction::Max,
                                    _ => unreachable!(),
                                },
                                arg: fc.args[0].expression,
                                arg1: Some(fc.args[1].expression),
                                arg2: None,
                            }),
                            sampler: None,
                            statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
                        })
                    }
                    "mix" | "clamp" => {
                        if fc.args.len() != 3 {
                            return Err(ErrorKind::WrongNumberArgs(name, 3, fc.args.len()));
                        }
                        Ok(ExpressionRule {
                            expression: self.context.expressions.append(Expression::Math {
                                fun: match name.as_str() {
                                    "mix" => MathFunction::Mix,
                                    "clamp" => MathFunction::Clamp,
                                    _ => unreachable!(),
                                },
                                arg: fc.args[0].expression,
                                arg1: Some(fc.args[1].expression),
                                arg2: Some(fc.args[2].expression),
                            }),
                            sampler: None,
                            statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
                        })
                    }
                    "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
                    | "notEqual" => {
                        if fc.args.len() != 2 {
                            return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
                        }
                        Ok(ExpressionRule {
                            expression: self.context.expressions.append(Expression::Binary {
                                op: match name.as_str() {
                                    "lessThan" => BinaryOperator::Less,
                                    "greaterThan" => BinaryOperator::Greater,
                                    "lessThanEqual" => BinaryOperator::LessEqual,
                                    "greaterThanEqual" => BinaryOperator::GreaterEqual,
                                    "equal" => BinaryOperator::Equal,
                                    "notEqual" => BinaryOperator::NotEqual,
                                    _ => unreachable!(),
                                },
                                left: fc.args[0].expression,
                                right: fc.args[1].expression,
                            }),
                            sampler: None,
                            statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
                        })
                    }
                    "isinf" => self.parse_relational_fun(name, fc.args, RelationalFunction::IsInf),
                    "isnan" => self.parse_relational_fun(name, fc.args, RelationalFunction::IsNan),
                    "all" => self.parse_relational_fun(name, fc.args, RelationalFunction::All),
                    "any" => self.parse_relational_fun(name, fc.args, RelationalFunction::Any),
                    func_name => {
                        let function = *self.lookup_function.get(func_name).ok_or_else(|| {
                            ErrorKind::SemanticError(
                                format!("Unknown function: {}", func_name).into(),
                            )
                        })?;
                        let arguments: Vec<_> = fc.args.iter().map(|a| a.expression).collect();
                        let mut statements: Vec<_> =
                            fc.args.into_iter().flat_map(|a| a.statements).collect();
                        let expression =
                            self.context.expressions.append(Expression::Call(function));
                        statements.push(crate::Statement::Call {
                            function,
                            arguments,
                            result: Some(expression),
                        });
                        Ok(ExpressionRule {
                            expression,
                            sampler: None,
                            statements,
                        })
                    }
                }
            }
        }
    }

    pub fn parse_relational_fun(
        &mut self,
        name: String,
        args: Vec<ExpressionRule>,
        fun: RelationalFunction,
    ) -> Result<ExpressionRule, ErrorKind> {
        if args.len() != 1 {
            return Err(ErrorKind::WrongNumberArgs(name, 1, args.len()));
        }
        Ok(ExpressionRule {
            expression: self.context.expressions.append(Expression::Relational {
                fun,
                argument: args[0].expression,
            }),
            sampler: None,
            statements: args.into_iter().flat_map(|a| a.statements).collect(),
        })
    }

    pub fn add_function_prelude(&mut self) {
        for (var_handle, var) in self.module.global_variables.iter() {
            if let Some(name) = var.name.as_ref() {
                let expr = self
                    .context
                    .expressions
                    .append(Expression::GlobalVariable(var_handle));
                self.context
                    .lookup_global_var_exps
                    .insert(name.clone(), expr);
            } else {
                let ty = &self.module.types[var.ty];
                // anonymous structs
                if let TypeInner::Struct { ref members, .. } = ty.inner {
                    let base = self
                        .context
                        .expressions
                        .append(Expression::GlobalVariable(var_handle));
                    for (idx, member) in members.iter().enumerate() {
                        if let Some(name) = member.name.as_ref() {
                            let exp = self.context.expressions.append(Expression::AccessIndex {
                                base,
                                index: idx as u32,
                            });
                            self.context
                                .lookup_global_var_exps
                                .insert(name.clone(), exp);
                        }
                    }
                }
            }
        }

        for (handle, constant) in self.module.constants.iter() {
            if let Some(name) = constant.name.as_ref() {
                let expr = self
                    .context
                    .expressions
                    .append(Expression::Constant(handle));
                self.context.lookup_constant_exps.insert(name.clone(), expr);
            }
        }
    }

    pub fn function_definition(&mut self, mut f: Function, mut block: Block) -> Function {
        std::mem::swap(&mut f.expressions, &mut self.context.expressions);
        std::mem::swap(&mut f.local_variables, &mut self.context.local_variables);
        std::mem::swap(&mut f.arguments, &mut self.context.arguments);
        self.context.clear_scopes();
        self.context.lookup_global_var_exps.clear();
        self.context.typifier = Typifier::new();
        ensure_block_returns(&mut block);
        f.body = block;
        f
    }

    pub fn declare_function(&mut self, f: Function) -> Result<(), ErrorKind> {
        let name = f
            .name
            .clone()
            .ok_or_else(|| ErrorKind::SemanticError("Unnamed function".into()))?;
        let stage = self.entry_points.get(&name);
        if let Some(&stage) = stage {
            self.module.entry_points.push(EntryPoint {
                name,
                stage,
                early_depth_test: None,
                workgroup_size: [0; 3], //TODO
                function: f,
            });
        } else {
            let handle = self.module.functions.append(f);
            self.lookup_function.insert(name, handle);
        }
        Ok(())
    }
}
