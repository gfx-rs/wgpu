use crate::{
    front::glsl::{
        ast::ParameterQualifier, context::Context, parser::ParsingContext, token::TokenValue,
        variables::VarDeclaration, ErrorKind, Parser, Result, Token,
    },
    Block, ConstantInner, Expression, ScalarValue, Statement, SwitchCase, UnaryOperator,
};

impl<'source> ParsingContext<'source> {
    pub fn peek_parameter_qualifier(&mut self, parser: &mut Parser) -> bool {
        self.peek(parser).map_or(false, |t| match t.value {
            TokenValue::In | TokenValue::Out | TokenValue::InOut | TokenValue::Const => true,
            _ => false,
        })
    }

    /// Returns the parsed `ParameterQualifier` or `ParameterQualifier::In`
    pub fn parse_parameter_qualifier(&mut self, parser: &mut Parser) -> ParameterQualifier {
        if self.peek_parameter_qualifier(parser) {
            match self.bump(parser).unwrap().value {
                TokenValue::In => ParameterQualifier::In,
                TokenValue::Out => ParameterQualifier::Out,
                TokenValue::InOut => ParameterQualifier::InOut,
                TokenValue::Const => ParameterQualifier::Const,
                _ => unreachable!(),
            }
        } else {
            ParameterQualifier::In
        }
    }

    pub fn parse_statement(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
    ) -> Result<()> {
        // TODO: This prevents snippets like the following from working
        // ```glsl
        // vec4(1.0);
        // ```
        // But this would require us to add lookahead to also support
        // declarations and since this statement is very unlikely and most
        // likely an error, for now we don't support it
        if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
            self.parse_declaration(parser, ctx, body, false)?;
            return Ok(());
        }

        match self.expect_peek(parser)?.value {
            TokenValue::Continue => {
                self.bump(parser)?;
                body.push(Statement::Continue);
                self.expect(parser, TokenValue::Semicolon)?;
            }
            TokenValue::Break => {
                self.bump(parser)?;
                body.push(Statement::Break);
                self.expect(parser, TokenValue::Semicolon)?;
            }
            TokenValue::Return => {
                self.bump(parser)?;
                let value = match self.expect_peek(parser)?.value {
                    TokenValue::Semicolon => {
                        self.bump(parser)?;
                        None
                    }
                    _ => {
                        // TODO: Implicit conversions
                        let expr = self.parse_expression(parser, ctx, body)?;
                        self.expect(parser, TokenValue::Semicolon)?;
                        Some(ctx.lower_expect(parser, expr, false, body)?.0)
                    }
                };

                ctx.emit_flush(body);
                ctx.emit_start();

                body.push(Statement::Return { value })
            }
            TokenValue::Discard => {
                self.bump(parser)?;
                body.push(Statement::Kill);
                self.expect(parser, TokenValue::Semicolon)?;
            }
            TokenValue::If => {
                self.bump(parser)?;

                self.expect(parser, TokenValue::LeftParen)?;
                let condition = {
                    let expr = self.parse_expression(parser, ctx, body)?;
                    ctx.lower_expect(parser, expr, false, body)?.0
                };
                self.expect(parser, TokenValue::RightParen)?;

                ctx.emit_flush(body);
                ctx.emit_start();

                let mut accept = Block::new();
                self.parse_statement(parser, ctx, &mut accept)?;

                let mut reject = Block::new();
                if self.bump_if(parser, TokenValue::Else).is_some() {
                    self.parse_statement(parser, ctx, &mut reject)?;
                }

                body.push(Statement::If {
                    condition,
                    accept,
                    reject,
                });
            }
            TokenValue::Switch => {
                self.bump(parser)?;

                self.expect(parser, TokenValue::LeftParen)?;
                // TODO: Implicit conversions
                let selector = {
                    let expr = self.parse_expression(parser, ctx, body)?;
                    ctx.lower_expect(parser, expr, false, body)?.0
                };
                self.expect(parser, TokenValue::RightParen)?;

                ctx.emit_flush(body);
                ctx.emit_start();

                let mut cases = Vec::new();
                let mut default = Block::new();

                self.expect(parser, TokenValue::LeftBrace)?;
                loop {
                    match self.expect_peek(parser)?.value {
                        TokenValue::Case => {
                            self.bump(parser)?;
                            let value = {
                                let expr = self.parse_expression(parser, ctx, body)?;
                                let (root, meta) = ctx.lower_expect(parser, expr, false, body)?;
                                let constant = parser.solve_constant(ctx, root, meta)?;

                                match parser.module.constants[constant].inner {
                                    ConstantInner::Scalar {
                                        value: ScalarValue::Sint(int),
                                        ..
                                    } => int as i32,
                                    ConstantInner::Scalar {
                                        value: ScalarValue::Uint(int),
                                        ..
                                    } => int as i32,
                                    _ => {
                                        return Err(ErrorKind::SemanticError(
                                            meta,
                                            "Case values can only be integers".into(),
                                        ))
                                    }
                                }
                            };

                            self.expect(parser, TokenValue::Colon)?;

                            let mut body = Block::new();

                            loop {
                                match self.expect_peek(parser)?.value {
                                    TokenValue::Case
                                    | TokenValue::Default
                                    | TokenValue::RightBrace => break,
                                    _ => self.parse_statement(parser, ctx, &mut body)?,
                                }
                            }

                            let mut fall_through = true;

                            for (i, stmt) in body.iter().enumerate() {
                                if let Statement::Break = *stmt {
                                    fall_through = false;
                                    body.drain(i..);
                                    break;
                                }
                            }

                            cases.push(SwitchCase {
                                value,
                                body,
                                fall_through,
                            })
                        }
                        TokenValue::Default => {
                            let Token { meta, .. } = self.bump(parser)?;
                            self.expect(parser, TokenValue::Colon)?;

                            if !default.is_empty() {
                                return Err(ErrorKind::SemanticError(
                                    meta,
                                    "Can only have one default case per switch statement".into(),
                                ));
                            }

                            loop {
                                match self.expect_peek(parser)?.value {
                                    TokenValue::Case | TokenValue::RightBrace => break,
                                    _ => self.parse_statement(parser, ctx, &mut default)?,
                                }
                            }

                            for (i, stmt) in default.iter().enumerate() {
                                if let Statement::Break = *stmt {
                                    default.drain(i..);
                                    break;
                                }
                            }
                        }
                        TokenValue::RightBrace => {
                            self.bump(parser)?;
                            break;
                        }
                        _ => {
                            return Err(ErrorKind::InvalidToken(
                                self.bump(parser)?,
                                vec![
                                    TokenValue::Case.into(),
                                    TokenValue::Default.into(),
                                    TokenValue::RightBrace.into(),
                                ],
                            ))
                        }
                    }
                }

                body.push(Statement::Switch {
                    selector,
                    cases,
                    default,
                });
            }
            TokenValue::While => {
                self.bump(parser)?;

                let mut loop_body = Block::new();

                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut loop_body)?;
                self.expect(parser, TokenValue::RightParen)?;

                let expr = ctx.lower_expect(parser, root, false, &mut loop_body)?.0;
                let condition = ctx.add_expression(
                    Expression::Unary {
                        op: UnaryOperator::Not,
                        expr,
                    },
                    &mut loop_body,
                );

                ctx.emit_flush(&mut loop_body);
                ctx.emit_start();

                loop_body.push(Statement::If {
                    condition,
                    accept: vec![Statement::Break],
                    reject: Block::new(),
                });

                self.parse_statement(parser, ctx, &mut loop_body)?;

                body.push(Statement::Loop {
                    body: loop_body,
                    continuing: Block::new(),
                })
            }
            TokenValue::Do => {
                self.bump(parser)?;

                let mut loop_body = Block::new();
                self.parse_statement(parser, ctx, &mut loop_body)?;

                self.expect(parser, TokenValue::While)?;
                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut loop_body)?;
                self.expect(parser, TokenValue::RightParen)?;

                let expr = ctx.lower_expect(parser, root, false, &mut loop_body)?.0;
                let condition = ctx.add_expression(
                    Expression::Unary {
                        op: UnaryOperator::Not,
                        expr,
                    },
                    &mut loop_body,
                );

                ctx.emit_flush(&mut loop_body);
                ctx.emit_start();

                loop_body.push(Statement::If {
                    condition,
                    accept: vec![Statement::Break],
                    reject: Block::new(),
                });

                body.push(Statement::Loop {
                    body: loop_body,
                    continuing: Block::new(),
                })
            }
            TokenValue::For => {
                self.bump(parser)?;

                ctx.push_scope();
                self.expect(parser, TokenValue::LeftParen)?;

                if self.bump_if(parser, TokenValue::Semicolon).is_none() {
                    if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
                        self.parse_declaration(parser, ctx, body, false)?;
                    } else {
                        let expr = self.parse_expression(parser, ctx, body)?;
                        ctx.lower(parser, expr, false, body)?;
                        self.expect(parser, TokenValue::Semicolon)?;
                    }
                }

                let (mut block, mut continuing) = (Block::new(), Block::new());

                if self.bump_if(parser, TokenValue::Semicolon).is_none() {
                    let expr = if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
                        let qualifiers = self.parse_type_qualifiers(parser)?;
                        let (ty, meta) = self.parse_type_non_void(parser)?;
                        let name = self.expect_ident(parser)?.0;

                        self.expect(parser, TokenValue::Assign)?;

                        let (value, end_meta) =
                            self.parse_initializer(parser, ty, ctx, &mut block)?;

                        let decl = VarDeclaration {
                            qualifiers: &qualifiers,
                            ty,
                            name: Some(name),
                            init: None,
                            meta: meta.union(&end_meta),
                        };

                        let pointer = parser.add_local_var(ctx, &mut block, decl)?;

                        ctx.emit_flush(&mut block);
                        ctx.emit_start();

                        block.push(Statement::Store { pointer, value });

                        value
                    } else {
                        let root = self.parse_expression(parser, ctx, &mut block)?;
                        ctx.lower_expect(parser, root, false, &mut block)?.0
                    };

                    let condition = ctx.add_expression(
                        Expression::Unary {
                            op: UnaryOperator::Not,
                            expr,
                        },
                        &mut block,
                    );

                    ctx.emit_flush(&mut block);
                    ctx.emit_start();

                    block.push(Statement::If {
                        condition,
                        accept: vec![Statement::Break],
                        reject: Block::new(),
                    });

                    self.expect(parser, TokenValue::Semicolon)?;
                }

                match self.expect_peek(parser)?.value {
                    TokenValue::RightParen => {}
                    _ => {
                        let rest = self.parse_expression(parser, ctx, &mut continuing)?;
                        ctx.lower(parser, rest, false, &mut continuing)?;
                    }
                }

                self.expect(parser, TokenValue::RightParen)?;

                self.parse_statement(parser, ctx, &mut block)?;

                body.push(Statement::Loop {
                    body: block,
                    continuing,
                });

                ctx.remove_current_scope();
            }
            TokenValue::LeftBrace => {
                self.bump(parser)?;

                let mut block = Block::new();
                ctx.push_scope();

                self.parse_compound_statement(parser, ctx, &mut block)?;

                ctx.remove_current_scope();
                body.push(Statement::Block(block));
            }
            TokenValue::Plus
            | TokenValue::Dash
            | TokenValue::Bang
            | TokenValue::Tilde
            | TokenValue::LeftParen
            | TokenValue::Identifier(_)
            | TokenValue::TypeName(_)
            | TokenValue::IntConstant(_)
            | TokenValue::BoolConstant(_)
            | TokenValue::FloatConstant(_) => {
                let expr = self.parse_expression(parser, ctx, body)?;
                ctx.lower(parser, expr, false, body)?;
                self.expect(parser, TokenValue::Semicolon)?;
            }
            TokenValue::Semicolon => {
                self.bump(parser)?;
            }
            _ => {}
        }

        Ok(())
    }

    pub fn parse_compound_statement(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
    ) -> Result<()> {
        loop {
            if self.bump_if(parser, TokenValue::RightBrace).is_some() {
                break;
            }

            self.parse_statement(parser, ctx, body)?;
        }

        Ok(())
    }

    pub fn parse_function_args(
        &mut self,
        parser: &mut Parser,
        context: &mut Context,
        body: &mut Block,
    ) -> Result<()> {
        loop {
            if self.peek_type_name(parser) || self.peek_parameter_qualifier(parser) {
                let qualifier = self.parse_parameter_qualifier(parser);
                let ty = self.parse_type_non_void(parser)?.0;

                match self.expect_peek(parser)?.value {
                    TokenValue::Comma => {
                        self.bump(parser)?;
                        context.add_function_arg(parser, body, None, ty, qualifier);
                        continue;
                    }
                    TokenValue::Identifier(_) => {
                        let name = self.expect_ident(parser)?.0;

                        let size = self.parse_array_specifier(parser)?;
                        let ty = parser.maybe_array(ty, size);

                        context.add_function_arg(parser, body, Some(name), ty, qualifier);

                        if self.bump_if(parser, TokenValue::Comma).is_some() {
                            continue;
                        }

                        break;
                    }
                    _ => break,
                }
            }

            break;
        }

        Ok(())
    }
}
