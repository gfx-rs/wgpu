use crate::{Block, Expression, Function, SampleLevel, TypeInner, proc::{ensure_block_returns, Typifier}};

use super::{ast::*, error::ErrorKind};

impl Program {
    pub fn function_call(&mut self, fc: FunctionCall) -> Result<ExpressionRule, ErrorKind> {
        match fc.kind {
            FunctionCallKind::TypeConstructor(ty) => {
                let h = if fc.args.len() == 1 {
                    let kind = self.module.types[ty].inner.scalar_kind().ok_or(
                        ErrorKind::SemanticError("Can only cast to scalar or vector"),
                    )?;
                    self.context.expressions.append(Expression::As {
                        kind,
                        expr: fc.args[0].expression,
                        convert: true,
                    })
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
                        //TODO: check args len
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
                        //TODO: check args len
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
                            return Err(ErrorKind::SemanticError("Bad call to texture"));
                        }
                    }
                    _ => {
                        return Err(ErrorKind::NotImplemented("Function call"));
                    }
                }
            }
        }
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
                if let TypeInner::Struct {
                    block: _,
                    ref members,
                } = ty.inner
                {
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
    }

    pub fn function_definition(&mut self, mut f: Function, mut block: Block) -> Function {
        std::mem::swap(&mut f.expressions, &mut self.context.expressions);
        std::mem::swap(&mut f.local_variables, &mut self.context.local_variables);
        self.context.clear_scopes();
        self.context.lookup_global_var_exps.clear();
        self.context.typifier = Typifier::new();
        ensure_block_returns(&mut block);
        f.body = block;
        f.fill_global_use(&self.module.global_variables);
        f
    }
}
