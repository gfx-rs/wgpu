use crate::{
    proc::ensure_block_returns, Arena, BinaryOperator, Binding, BuiltIn, EntryPoint, Expression,
    Function, FunctionArgument, FunctionResult, MathFunction, RelationalFunction, SampleLevel,
    ShaderStage, Statement, TypeInner,
};

use super::{ast::*, error::ErrorKind};

impl Program<'_> {
    // pub fn function_call(
    //     &mut self,
    //     context: &mut FunctionContext,
    //     fc: FunctionCall,
    // ) -> Result<ExpressionRule, ErrorKind> {
    //     match fc.kind {
    //         FunctionCallKind::TypeConstructor(ty) => {
    //             let h = if fc.args.len() == 1 {
    //                 let is_vec = match *self.resolve_type(context, fc.args[0].expression)? {
    //                     TypeInner::Vector { .. } => true,
    //                     _ => false,
    //                 };

    //                 match self.module.types[ty].inner {
    //                     TypeInner::Vector { size, .. } if !is_vec => {
    //                         context.function.expressions.append(Expression::Splat {
    //                             size,
    //                             value: fc.args[0].expression,
    //                         })
    //                     }
    //                     TypeInner::Scalar { kind, width }
    //                     | TypeInner::Vector { kind, width, .. } => {
    //                         context.function.expressions.append(Expression::As {
    //                             kind,
    //                             expr: fc.args[0].expression,
    //                             convert: Some(width),
    //                         })
    //                     }
    //                     TypeInner::Matrix {
    //                         columns,
    //                         rows,
    //                         width,
    //                     } => {
    //                         let value = context.function.expressions.append(Expression::As {
    //                             kind: ScalarKind::Float,
    //                             expr: fc.args[0].expression,
    //                             convert: Some(width),
    //                         });

    //                         let column = if is_vec {
    //                             value
    //                         } else {
    //                             context
    //                                 .function
    //                                 .expressions
    //                                 .append(Expression::Splat { size: rows, value })
    //                         };

    //                         let columns =
    //                             std::iter::repeat(column).take(columns as usize).collect();

    //             context.function.expressions.append(Expression::Compose {
    //                 ty,
    //                 components: columns,
    //             })
    //         }
    //         _ => return Err(ErrorKind::SemanticError("Bad cast".into())),
    //     }
    // } else {
    //     context.function.expressions.append(Expression::Compose {
    //         ty,
    //         components: fc.args.iter().map(|a| a.expression).collect(),
    //     })
    // };

    //             Ok(ExpressionRule {
    //                 expression: h,
    //                 statements: fc
    //                     .args
    //                     .into_iter()
    //                     .map(|a| a.statements)
    //                     .flatten()
    //                     .collect(),
    //                 sampler: None,
    //             })
    //         }
    //         FunctionCallKind::Function(name) => {
    //             match name.as_str() {
    //                 "sampler2D" => {
    //                     if fc.args.len() != 2 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
    //                     }
    //                     Ok(ExpressionRule {
    //                         expression: fc.args[0].expression,
    //                         sampler: Some(fc.args[1].expression),
    //                         statements: fc
    //                             .args
    //                             .into_iter()
    //                             .map(|a| a.statements)
    //                             .flatten()
    //                             .collect(),
    //                     })
    //                 }
    //                 "texture" => {
    //                     if fc.args.len() != 2 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
    //                     }
    //                     if let Some(sampler) = fc.args[0].sampler {
    //                         Ok(ExpressionRule {
    //                             expression: context.function.expressions.append(
    //                                 Expression::ImageSample {
    //                                     image: fc.args[0].expression,
    //                                     sampler,
    //                                     coordinate: fc.args[1].expression,
    //                                     array_index: None, //TODO
    //                                     offset: None,      //TODO
    //                                     level: SampleLevel::Auto,
    //                                     depth_ref: None,
    //                                 },
    //                             ),
    //                             sampler: None,
    //                             statements: fc
    //                                 .args
    //                                 .into_iter()
    //                                 .map(|a| a.statements)
    //                                 .flatten()
    //                                 .collect(),
    //                         })
    //                     } else {
    //                         Err(ErrorKind::SemanticError("Bad call to texture".into()))
    //                     }
    //                 }
    //                 "textureLod" => {
    //                     if fc.args.len() != 3 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 3, fc.args.len()));
    //                     }
    //                     if let Some(sampler) = fc.args[0].sampler {
    //                         Ok(ExpressionRule {
    //                             expression: context.function.expressions.append(
    //                                 Expression::ImageSample {
    //                                     image: fc.args[0].expression,
    //                                     sampler,
    //                                     coordinate: fc.args[1].expression,
    //                                     array_index: None, //TODO
    //                                     offset: None,      //TODO
    //                                     level: SampleLevel::Exact(fc.args[2].expression),
    //                                     depth_ref: None,
    //                                 },
    //                             ),
    //                             sampler: None,
    //                             statements: fc
    //                                 .args
    //                                 .into_iter()
    //                                 .map(|a| a.statements)
    //                                 .flatten()
    //                                 .collect(),
    //                         })
    //                     } else {
    //                         Err(ErrorKind::SemanticError("Bad call to textureLod".into()))
    //                     }
    //                 }
    //                 "ceil" | "round" | "floor" | "fract" | "trunc" | "sin" | "abs" | "sqrt"
    //                 | "inversesqrt" | "exp" | "exp2" | "sign" | "transpose" | "inverse"
    //                 | "normalize" => {
    //                     if fc.args.len() != 1 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 1, fc.args.len()));
    //                     }
    //                     Ok(ExpressionRule {
    //                         expression: context.function.expressions.append(Expression::Math {
    //                             fun: match name.as_str() {
    //                                 "ceil" => MathFunction::Ceil,
    //                                 "round" => MathFunction::Round,
    //                                 "floor" => MathFunction::Floor,
    //                                 "fract" => MathFunction::Fract,
    //                                 "trunc" => MathFunction::Trunc,
    //                                 "sin" => MathFunction::Sin,
    //                                 "abs" => MathFunction::Abs,
    //                                 "sqrt" => MathFunction::Sqrt,
    //                                 "inversesqrt" => MathFunction::InverseSqrt,
    //                                 "exp" => MathFunction::Exp,
    //                                 "exp2" => MathFunction::Exp2,
    //                                 "sign" => MathFunction::Sign,
    //                                 "transpose" => MathFunction::Transpose,
    //                                 "inverse" => MathFunction::Inverse,
    //                                 "normalize" => MathFunction::Normalize,
    //                                 _ => unreachable!(),
    //                             },
    //                             arg: fc.args[0].expression,
    //                             arg1: None,
    //                             arg2: None,
    //                         }),
    //                         sampler: None,
    //                         statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
    //                     })
    //                 }
    //                 "pow" | "dot" | "max" => {
    //                     if fc.args.len() != 2 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
    //                     }
    //                     Ok(ExpressionRule {
    //                         expression: context.function.expressions.append(Expression::Math {
    //                             fun: match name.as_str() {
    //                                 "pow" => MathFunction::Pow,
    //                                 "dot" => MathFunction::Dot,
    //                                 "max" => MathFunction::Max,
    //                                 _ => unreachable!(),
    //                             },
    //                             arg: fc.args[0].expression,
    //                             arg1: Some(fc.args[1].expression),
    //                             arg2: None,
    //                         }),
    //                         sampler: None,
    //                         statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
    //                     })
    //                 }
    //                 "mix" | "clamp" => {
    //                     if fc.args.len() != 3 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 3, fc.args.len()));
    //                     }
    //                     Ok(ExpressionRule {
    //                         expression: context.function.expressions.append(Expression::Math {
    //                             fun: match name.as_str() {
    //                                 "mix" => MathFunction::Mix,
    //                                 "clamp" => MathFunction::Clamp,
    //                                 _ => unreachable!(),
    //                             },
    //                             arg: fc.args[0].expression,
    //                             arg1: Some(fc.args[1].expression),
    //                             arg2: Some(fc.args[2].expression),
    //                         }),
    //                         sampler: None,
    //                         statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
    //                     })
    //                 }
    //                 "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
    //                 | "notEqual" => {
    //                     if fc.args.len() != 2 {
    //                         return Err(ErrorKind::WrongNumberArgs(name, 2, fc.args.len()));
    //                     }
    //                     Ok(ExpressionRule {
    //                         expression: context.function.expressions.append(Expression::Binary {
    //                             op: match name.as_str() {
    //                                 "lessThan" => BinaryOperator::Less,
    //                                 "greaterThan" => BinaryOperator::Greater,
    //                                 "lessThanEqual" => BinaryOperator::LessEqual,
    //                                 "greaterThanEqual" => BinaryOperator::GreaterEqual,
    //                                 "equal" => BinaryOperator::Equal,
    //                                 "notEqual" => BinaryOperator::NotEqual,
    //                                 _ => unreachable!(),
    //                             },
    //                             left: fc.args[0].expression,
    //                             right: fc.args[1].expression,
    //                         }),
    //                         sampler: None,
    //                         statements: fc.args.into_iter().flat_map(|a| a.statements).collect(),
    //                     })
    //                 }
    //                 "isinf" => {
    //                     self.parse_relational_fun(context, name, fc.args, RelationalFunction::IsInf)
    //                 }
    //                 "isnan" => {
    //                     self.parse_relational_fun(context, name, fc.args, RelationalFunction::IsNan)
    //                 }
    //                 "all" => {
    //                     self.parse_relational_fun(context, name, fc.args, RelationalFunction::All)
    //                 }
    //                 "any" => {
    //                     self.parse_relational_fun(context, name, fc.args, RelationalFunction::Any)
    //                 }
    //                 func_name => {
    //                     let function = *self.lookup_function.get(func_name).ok_or_else(|| {
    //                         ErrorKind::SemanticError(
    //                             format!("Unknown function: {}", func_name).into(),
    //                         )
    //                     })?;
    //                     let arguments: Vec<_> = fc.args.iter().map(|a| a.expression).collect();
    //                     let mut statements: Vec<_> =
    //                         fc.args.into_iter().flat_map(|a| a.statements).collect();
    //                     let expression = context
    //                         .function
    //                         .expressions
    //                         .append(Expression::Call(function));
    //                     statements.push(crate::Statement::Call {
    //                         function,
    //                         arguments,
    //                         result: Some(expression),
    //                     });
    //                     Ok(ExpressionRule {
    //                         expression,
    //                         sampler: None,
    //                         statements,
    //                     })
    //                 }
    //             }
    //         }
    //     }
    // }

    // TODO: Reenable later
    // pub fn parse_relational_fun(
    //     &mut self,
    //     context: &mut FunctionContext,
    //     name: String,
    //     args: Vec<ExpressionRule>,
    //     fun: RelationalFunction,
    // ) -> Result<ExpressionRule, ErrorKind> {
    //     if args.len() != 1 {
    //         return Err(ErrorKind::WrongNumberArgs(name, 1, args.len()));
    //     }
    //     Ok(ExpressionRule {
    //         expression: context.function.expressions.append(Expression::Relational {
    //             fun,
    //             argument: args[0].expression,
    //         }),
    //         sampler: None,
    //         statements: args.into_iter().flat_map(|a| a.statements).collect(),
    //     })
    // }

    pub fn add_function(&mut self, mut function: Function) -> Result<(), ErrorKind> {
        ensure_block_returns(&mut function.body);
        let name = function
            .name
            .clone()
            .ok_or_else(|| ErrorKind::SemanticError("Unnamed function".into()))?;
        let stage = self.entry_points.get(&name);

        // Add the input variables as a function argument
        function.arguments.push(FunctionArgument {
            name: None,
            ty: self.input_struct,
            binding: None,
        });

        let handle = self.module.functions.append(function);

        if let Some(&stage) = stage {
            self.entries.push((name, stage, handle));
        } else {
            self.lookup_function.insert(name, handle);
        }

        Ok(())
    }

    pub fn add_entry_points(&mut self) {
        for (name, stage, function) in self.entries.iter().cloned() {
            let mut arguments = Vec::new();
            let mut expressions = Arena::new();
            let mut body = Vec::new();

            for (built_in, handle) in self.built_ins.iter().copied() {
                let ty = self.module.global_variables[handle].ty;
                let arg = arguments.len() as u32;

                arguments.push(FunctionArgument {
                    name: None,
                    ty,
                    binding: Some(Binding::BuiltIn(built_in)),
                });

                let pointer = expressions.append(Expression::GlobalVariable(handle));
                let value = expressions.append(Expression::FunctionArgument(arg));

                body.push(Statement::Store { pointer, value });
            }

            let i = arguments.len() as u32;
            arguments.push(FunctionArgument {
                name: None,
                ty: self.input_struct,
                binding: None,
            });

            let res = expressions.append(Expression::Call(function));

            body.push(Statement::Call {
                function,
                arguments: vec![expressions.append(Expression::FunctionArgument(i))],
                result: Some(res),
            });

            for (i, (built_in, handle)) in self.built_ins.iter().copied().enumerate() {
                if !should_write(built_in, stage) {
                    continue;
                }

                let value = expressions.append(Expression::GlobalVariable(handle));
                let pointer = expressions.append(Expression::FunctionArgument(i as u32));

                body.push(Statement::Store { pointer, value });
            }

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
                    result: Some(FunctionResult {
                        ty: self.output_struct,
                        binding: None,
                    }),
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
