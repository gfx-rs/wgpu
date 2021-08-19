use crate::front::glsl::context::ExprPos;
use crate::front::glsl::SourceMetadata;
use crate::{
    front::glsl::{
        ast::ParameterQualifier,
        context::Context,
        parser::ParsingContext,
        token::{Token, TokenValue},
        variables::VarDeclaration,
        Error, ErrorKind, Parser, Result,
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
    ) -> Result<Option<SourceMetadata>> {
        // TODO: This prevents snippets like the following from working
        // ```glsl
        // vec4(1.0);
        // ```
        // But this would require us to add lookahead to also support
        // declarations and since this statement is very unlikely and most
        // likely an error, for now we don't support it
        if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
            return self.parse_declaration(parser, ctx, body, false);
        }

        let new_break = || {
            let mut block = Block::new();
            block.push(Statement::Break, crate::Span::Unknown);
            block
        };

        let &Token { ref value, meta } = self.expect_peek(parser)?;

        let meta_rest = match *value {
            TokenValue::Continue => {
                body.push(Statement::Continue, self.bump(parser)?.meta.as_span());
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
            TokenValue::Break => {
                body.push(Statement::Break, self.bump(parser)?.meta.as_span());
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
            TokenValue::Return => {
                self.bump(parser)?;
                let (value, meta) = match self.expect_peek(parser)?.value {
                    TokenValue::Semicolon => (None, self.bump(parser)?.meta),
                    _ => {
                        // TODO: Implicit conversions
                        let mut stmt = ctx.stmt_ctx();
                        let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                        self.expect(parser, TokenValue::Semicolon)?;
                        let (handle, meta) =
                            ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?;
                        (Some(handle), meta)
                    }
                };

                ctx.emit_flush(body);
                ctx.emit_start();

                body.push(Statement::Return { value }, meta.as_span());
                meta
            }
            TokenValue::Discard => {
                body.push(Statement::Kill, self.bump(parser)?.meta.as_span());
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
            TokenValue::If => {
                let mut meta = self.bump(parser)?.meta;

                self.expect(parser, TokenValue::LeftParen)?;
                let condition = {
                    let mut stmt = ctx.stmt_ctx();
                    let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                    let (handle, more_meta) =
                        ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?;
                    meta = meta.union(&more_meta);
                    handle
                };
                self.expect(parser, TokenValue::RightParen)?;

                ctx.emit_flush(body);
                ctx.emit_start();

                let mut accept = Block::new();
                if let Some(more_meta) = self.parse_statement(parser, ctx, &mut accept)? {
                    meta = meta.union(&more_meta)
                }

                let mut reject = Block::new();
                if self.bump_if(parser, TokenValue::Else).is_some() {
                    if let Some(more_meta) = self.parse_statement(parser, ctx, &mut reject)? {
                        meta = meta.union(&more_meta);
                    }
                }

                body.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    meta.as_span(),
                );
                meta
            }
            TokenValue::Switch => {
                let start_meta = self.bump(parser)?.meta;
                let end_meta;

                self.expect(parser, TokenValue::LeftParen)?;
                // TODO: Implicit conversions
                let selector = {
                    let mut stmt = ctx.stmt_ctx();
                    let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                    ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?.0
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
                                let mut stmt = ctx.stmt_ctx();
                                let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                                let (root, meta) =
                                    ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?;
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
                                        parser.errors.push(Error {
                                            kind: ErrorKind::SemanticError(
                                                "Case values can only be integers".into(),
                                            ),
                                            meta,
                                        });

                                        0
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
                                    _ => {
                                        self.parse_statement(parser, ctx, &mut body)?;
                                    }
                                }
                            }

                            let mut fall_through = true;

                            for (i, stmt) in body.iter().enumerate() {
                                if let Statement::Break = *stmt {
                                    fall_through = false;
                                    body.cull(i..);
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
                                parser.errors.push(Error {
                                    kind: ErrorKind::SemanticError(
                                        "Can only have one default case per switch statement"
                                            .into(),
                                    ),
                                    meta,
                                });
                            }

                            loop {
                                match self.expect_peek(parser)?.value {
                                    TokenValue::Case | TokenValue::RightBrace => break,
                                    _ => {
                                        self.parse_statement(parser, ctx, &mut default)?;
                                    }
                                }
                            }

                            for (i, stmt) in default.iter().enumerate() {
                                if let Statement::Break = *stmt {
                                    default.cull(i..);
                                    break;
                                }
                            }
                        }
                        TokenValue::RightBrace => {
                            end_meta = self.bump(parser)?.meta;
                            break;
                        }
                        _ => {
                            let Token { value, meta } = self.bump(parser)?;
                            return Err(Error {
                                kind: ErrorKind::InvalidToken(
                                    value,
                                    vec![
                                        TokenValue::Case.into(),
                                        TokenValue::Default.into(),
                                        TokenValue::RightBrace.into(),
                                    ],
                                ),
                                meta,
                            });
                        }
                    }
                }

                let meta = start_meta.union(&end_meta);
                body.push(
                    Statement::Switch {
                        selector,
                        cases,
                        default,
                    },
                    meta.as_span(),
                );
                meta
            }
            TokenValue::While => {
                let meta = self.bump(parser)?.meta;

                let mut loop_body = Block::new();

                let mut stmt = ctx.stmt_ctx();
                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut stmt, &mut loop_body)?;
                let meta = meta.union(&self.expect(parser, TokenValue::RightParen)?.meta);

                let (expr, expr_meta) =
                    ctx.lower_expect(stmt, parser, root, ExprPos::Rhs, &mut loop_body)?;
                let condition = ctx.add_expression(
                    Expression::Unary {
                        op: UnaryOperator::Not,
                        expr,
                    },
                    expr_meta,
                    &mut loop_body,
                );

                ctx.emit_flush(&mut loop_body);
                ctx.emit_start();

                loop_body.push(
                    Statement::If {
                        condition,
                        accept: new_break(),
                        reject: Block::new(),
                    },
                    crate::Span::Unknown,
                );

                let mut meta = meta.union(&expr_meta);

                if let Some(body_meta) = self.parse_statement(parser, ctx, &mut loop_body)? {
                    meta = meta.union(&body_meta);
                }

                body.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing: Block::new(),
                    },
                    meta.as_span(),
                );
                meta
            }
            TokenValue::Do => {
                let start_meta = self.bump(parser)?.meta;

                let mut loop_body = Block::new();
                self.parse_statement(parser, ctx, &mut loop_body)?;

                let mut stmt = ctx.stmt_ctx();

                self.expect(parser, TokenValue::While)?;
                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut stmt, &mut loop_body)?;
                let end_meta = self.expect(parser, TokenValue::RightParen)?.meta;

                let meta = start_meta.union(&end_meta);

                let (expr, expr_meta) =
                    ctx.lower_expect(stmt, parser, root, ExprPos::Rhs, &mut loop_body)?;
                let condition = ctx.add_expression(
                    Expression::Unary {
                        op: UnaryOperator::Not,
                        expr,
                    },
                    expr_meta,
                    &mut loop_body,
                );

                ctx.emit_flush(&mut loop_body);
                ctx.emit_start();

                loop_body.push(
                    Statement::If {
                        condition,
                        accept: new_break(),
                        reject: Block::new(),
                    },
                    crate::Span::Unknown,
                );

                body.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing: Block::new(),
                    },
                    meta.as_span(),
                );
                meta
            }
            TokenValue::For => {
                let meta = self.bump(parser)?.meta;

                ctx.push_scope();
                self.expect(parser, TokenValue::LeftParen)?;

                if self.bump_if(parser, TokenValue::Semicolon).is_none() {
                    if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
                        self.parse_declaration(parser, ctx, body, false)?;
                    } else {
                        let mut stmt = ctx.stmt_ctx();
                        let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                        ctx.lower(stmt, parser, expr, ExprPos::Rhs, body)?;
                        self.expect(parser, TokenValue::Semicolon)?;
                    }
                }

                let (mut block, mut continuing) = (Block::new(), Block::new());

                if self.bump_if(parser, TokenValue::Semicolon).is_none() {
                    let (expr, expr_meta) =
                        if self.peek_type_name(parser) || self.peek_type_qualifier(parser) {
                            let qualifiers = self.parse_type_qualifiers(parser)?;
                            let (ty, meta) = self.parse_type_non_void(parser)?;
                            let name = self.expect_ident(parser)?.0;

                            self.expect(parser, TokenValue::Assign)?;

                            let (value, end_meta) =
                                self.parse_initializer(parser, ty, ctx, &mut block)?;
                            let meta = meta.union(&end_meta);

                            let decl = VarDeclaration {
                                qualifiers: &qualifiers,
                                ty,
                                name: Some(name),
                                init: None,
                                meta,
                            };

                            let pointer = parser.add_local_var(ctx, &mut block, decl)?;

                            ctx.emit_flush(&mut block);
                            ctx.emit_start();

                            block.push(Statement::Store { pointer, value }, meta.as_span());

                            (value, end_meta)
                        } else {
                            let mut stmt = ctx.stmt_ctx();
                            let root = self.parse_expression(parser, ctx, &mut stmt, &mut block)?;
                            ctx.lower_expect(stmt, parser, root, ExprPos::Rhs, &mut block)?
                        };

                    let condition = ctx.add_expression(
                        Expression::Unary {
                            op: UnaryOperator::Not,
                            expr,
                        },
                        expr_meta,
                        &mut block,
                    );

                    ctx.emit_flush(&mut block);
                    ctx.emit_start();

                    block.push(
                        Statement::If {
                            condition,
                            accept: new_break(),
                            reject: Block::new(),
                        },
                        crate::Span::Unknown,
                    );

                    self.expect(parser, TokenValue::Semicolon)?;
                }

                match self.expect_peek(parser)?.value {
                    TokenValue::RightParen => {}
                    _ => {
                        let mut stmt = ctx.stmt_ctx();
                        let rest =
                            self.parse_expression(parser, ctx, &mut stmt, &mut continuing)?;
                        ctx.lower(stmt, parser, rest, ExprPos::Rhs, &mut continuing)?;
                    }
                }

                let mut meta = meta.union(&self.expect(parser, TokenValue::RightParen)?.meta);

                if let Some(stmt_meta) = self.parse_statement(parser, ctx, &mut block)? {
                    meta = meta.union(&stmt_meta);
                }

                body.push(
                    Statement::Loop {
                        body: block,
                        continuing,
                    },
                    meta.as_span(),
                );

                ctx.remove_current_scope();

                meta
            }
            TokenValue::LeftBrace => {
                let meta = self.bump(parser)?.meta;

                let mut block = Block::new();
                ctx.push_scope();

                let meta = self.parse_compound_statement(meta, parser, ctx, &mut block)?;

                ctx.remove_current_scope();
                body.push(Statement::Block(block), meta.as_span());

                meta
            }
            TokenValue::Plus
            | TokenValue::Dash
            | TokenValue::Bang
            | TokenValue::Tilde
            | TokenValue::LeftParen
            | TokenValue::Increment
            | TokenValue::Decrement
            | TokenValue::Identifier(_)
            | TokenValue::TypeName(_)
            | TokenValue::IntConstant(_)
            | TokenValue::BoolConstant(_)
            | TokenValue::FloatConstant(_) => {
                let mut stmt = ctx.stmt_ctx();
                let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                ctx.lower(stmt, parser, expr, ExprPos::Rhs, body)?;
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
            TokenValue::Semicolon => self.bump(parser)?.meta,
            _ => meta,
        };

        Ok(Some(meta.union(&meta_rest)))
    }

    pub fn parse_compound_statement(
        &mut self,
        mut meta: SourceMetadata,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
    ) -> Result<SourceMetadata> {
        loop {
            if let Some(Token {
                meta: brace_meta, ..
            }) = self.bump_if(parser, TokenValue::RightBrace)
            {
                meta = meta.union(&brace_meta);
                break;
            }

            if let Some(stmt_meta) = self.parse_statement(parser, ctx, body)? {
                meta = meta.union(&stmt_meta);
            }
        }

        Ok(meta)
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
                        let name_meta = self.expect_ident(parser)?;

                        let array_specifier = self.parse_array_specifier(parser)?;
                        let ty = parser.maybe_array(ty, name_meta.1, array_specifier);

                        context.add_function_arg(parser, body, Some(name_meta), ty, qualifier);

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
