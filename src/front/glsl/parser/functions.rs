use crate::front::glsl::context::ExprPos;
use crate::front::glsl::Span;
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
        terminator: &mut Option<usize>,
    ) -> Result<Option<Span>> {
        // Type qualifiers always identify a declaration statement
        if self.peek_type_qualifier(parser) {
            return self.parse_declaration(parser, ctx, body, false);
        }

        // Type names can identify either declaration statements or type constructors
        // depending on wether the token following the type name is a `(` (LeftParen)
        if self.peek_type_name(parser) {
            // Start by consuming the type name so that we can peek the token after it
            let token = self.bump(parser)?;
            // Peek the next token and check if it's a `(` (LeftParen) if so the statement
            // is a constructor, otherwise it's a declaration. We need to do the check
            // beforehand and not in the if since we will backtrack before the if
            let declaration = TokenValue::LeftParen != self.expect_peek(parser)?.value;

            self.backtrack(token)?;

            if declaration {
                return self.parse_declaration(parser, ctx, body, false);
            }
        }

        let new_break = || {
            let mut block = Block::new();
            block.push(Statement::Break, crate::Span::default());
            block
        };

        let &Token {
            ref value,
            mut meta,
        } = self.expect_peek(parser)?;

        let meta_rest = match *value {
            TokenValue::Continue => {
                let meta = self.bump(parser)?.meta;
                body.push(Statement::Continue, meta);
                terminator.get_or_insert(body.len());
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
            TokenValue::Break => {
                let meta = self.bump(parser)?.meta;
                body.push(Statement::Break, meta);
                terminator.get_or_insert(body.len());
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

                ctx.emit_restart(body);

                body.push(Statement::Return { value }, meta);
                terminator.get_or_insert(body.len());

                meta
            }
            TokenValue::Discard => {
                let meta = self.bump(parser)?.meta;
                body.push(Statement::Kill, meta);
                terminator.get_or_insert(body.len());

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
                    meta.subsume(more_meta);
                    handle
                };
                self.expect(parser, TokenValue::RightParen)?;

                ctx.emit_restart(body);

                let mut accept = Block::new();
                if let Some(more_meta) =
                    self.parse_statement(parser, ctx, &mut accept, &mut None)?
                {
                    meta.subsume(more_meta)
                }

                let mut reject = Block::new();
                if self.bump_if(parser, TokenValue::Else).is_some() {
                    if let Some(more_meta) =
                        self.parse_statement(parser, ctx, &mut reject, &mut None)?
                    {
                        meta.subsume(more_meta);
                    }
                }

                body.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    meta,
                );

                meta
            }
            TokenValue::Switch => {
                let mut meta = self.bump(parser)?.meta;
                let end_meta;

                self.expect(parser, TokenValue::LeftParen)?;

                let selector = {
                    let mut stmt = ctx.stmt_ctx();
                    let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                    ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?.0
                };

                self.expect(parser, TokenValue::RightParen)?;

                ctx.emit_restart(body);

                let mut cases = Vec::new();
                // Track if any default case is present in the switch statement.
                let mut default_present = false;

                self.expect(parser, TokenValue::LeftBrace)?;
                loop {
                    let value = match self.expect_peek(parser)?.value {
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
                            crate::SwitchValue::Integer(value)
                        }
                        TokenValue::Default => {
                            self.bump(parser)?;
                            default_present = true;
                            crate::SwitchValue::Default
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
                    };

                    self.expect(parser, TokenValue::Colon)?;

                    let mut body = Block::new();

                    let mut case_terminator = None;
                    loop {
                        match self.expect_peek(parser)?.value {
                            TokenValue::Case | TokenValue::Default | TokenValue::RightBrace => {
                                break
                            }
                            _ => {
                                self.parse_statement(parser, ctx, &mut body, &mut case_terminator)?;
                            }
                        }
                    }

                    let mut fall_through = true;

                    if let Some(mut idx) = case_terminator {
                        if let Statement::Break = body[idx - 1] {
                            fall_through = false;
                            idx -= 1;
                        }

                        body.cull(idx..)
                    }

                    cases.push(SwitchCase {
                        value,
                        body,
                        fall_through,
                    })
                }

                meta.subsume(end_meta);

                // NOTE: do not unwrap here since a switch statement isn't required
                // to have any cases.
                if let Some(case) = cases.last_mut() {
                    // GLSL requires that the last case not be empty, so we check
                    // that here and produce an error otherwise (fall_trough must
                    // also be checked because `break`s count as statements but
                    // they aren't added to the body)
                    if case.body.is_empty() && case.fall_through {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "last case/default label must be followed by statements".into(),
                            ),
                            meta,
                        })
                    }

                    // GLSL allows the last case to not have any `break` statement,
                    // this would mark it as fall trough but naga's IR requires that
                    // the last case must not be fall trough, so we mark need to mark
                    // the last case as not fall trough always.
                    case.fall_through = false;
                }

                // Add an empty default case in case non was present, this is needed because
                // naga's IR requires that all switch statements must have a default case but
                // GLSL doesn't require that, so we might need to add an empty default case.
                if !default_present {
                    cases.push(SwitchCase {
                        value: crate::SwitchValue::Default,
                        body: Block::new(),
                        fall_through: false,
                    })
                }

                body.push(Statement::Switch { selector, cases }, meta);

                meta
            }
            TokenValue::While => {
                let mut meta = self.bump(parser)?.meta;

                let mut loop_body = Block::new();

                let mut stmt = ctx.stmt_ctx();
                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut stmt, &mut loop_body)?;
                meta.subsume(self.expect(parser, TokenValue::RightParen)?.meta);

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

                ctx.emit_restart(&mut loop_body);

                loop_body.push(
                    Statement::If {
                        condition,
                        accept: new_break(),
                        reject: Block::new(),
                    },
                    crate::Span::default(),
                );

                meta.subsume(expr_meta);

                if let Some(body_meta) =
                    self.parse_statement(parser, ctx, &mut loop_body, &mut None)?
                {
                    meta.subsume(body_meta);
                }

                body.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing: Block::new(),
                        break_if: None,
                    },
                    meta,
                );

                meta
            }
            TokenValue::Do => {
                let mut meta = self.bump(parser)?.meta;

                let mut loop_body = Block::new();

                let mut terminator = None;
                self.parse_statement(parser, ctx, &mut loop_body, &mut terminator)?;

                let mut stmt = ctx.stmt_ctx();

                self.expect(parser, TokenValue::While)?;
                self.expect(parser, TokenValue::LeftParen)?;
                let root = self.parse_expression(parser, ctx, &mut stmt, &mut loop_body)?;
                let end_meta = self.expect(parser, TokenValue::RightParen)?.meta;

                meta.subsume(end_meta);

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

                ctx.emit_restart(&mut loop_body);

                loop_body.push(
                    Statement::If {
                        condition,
                        accept: new_break(),
                        reject: Block::new(),
                    },
                    crate::Span::default(),
                );

                if let Some(idx) = terminator {
                    loop_body.cull(idx..)
                }

                body.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing: Block::new(),
                        break_if: None,
                    },
                    meta,
                );

                meta
            }
            TokenValue::For => {
                let mut meta = self.bump(parser)?.meta;

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
                            let mut qualifiers = self.parse_type_qualifiers(parser)?;
                            let (ty, mut meta) = self.parse_type_non_void(parser)?;
                            let name = self.expect_ident(parser)?.0;

                            self.expect(parser, TokenValue::Assign)?;

                            let (value, end_meta) =
                                self.parse_initializer(parser, ty, ctx, &mut block)?;
                            meta.subsume(end_meta);

                            let decl = VarDeclaration {
                                qualifiers: &mut qualifiers,
                                ty,
                                name: Some(name),
                                init: None,
                                meta,
                            };

                            let pointer = parser.add_local_var(ctx, &mut block, decl)?;

                            ctx.emit_restart(&mut block);

                            block.push(Statement::Store { pointer, value }, meta);

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

                    ctx.emit_restart(&mut block);

                    block.push(
                        Statement::If {
                            condition,
                            accept: new_break(),
                            reject: Block::new(),
                        },
                        crate::Span::default(),
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

                meta.subsume(self.expect(parser, TokenValue::RightParen)?.meta);

                if let Some(stmt_meta) = self.parse_statement(parser, ctx, &mut block, &mut None)? {
                    meta.subsume(stmt_meta);
                }

                body.push(
                    Statement::Loop {
                        body: block,
                        continuing,
                        break_if: None,
                    },
                    meta,
                );

                ctx.remove_current_scope();

                meta
            }
            TokenValue::LeftBrace => {
                let meta = self.bump(parser)?.meta;

                let mut block = Block::new();
                ctx.push_scope();

                let mut block_terminator = None;
                let meta = self.parse_compound_statement(
                    meta,
                    parser,
                    ctx,
                    &mut block,
                    &mut block_terminator,
                )?;

                ctx.remove_current_scope();

                body.push(Statement::Block(block), meta);
                if block_terminator.is_some() {
                    terminator.get_or_insert(body.len());
                }

                meta
            }
            TokenValue::Semicolon => self.bump(parser)?.meta,
            _ => {
                // Attempt to force expression parsing for remainder of the
                // tokens. Unknown or invalid tokens will be caught there and
                // turned into an error.
                let mut stmt = ctx.stmt_ctx();
                let expr = self.parse_expression(parser, ctx, &mut stmt, body)?;
                ctx.lower(stmt, parser, expr, ExprPos::Rhs, body)?;
                self.expect(parser, TokenValue::Semicolon)?.meta
            }
        };

        meta.subsume(meta_rest);
        Ok(Some(meta))
    }

    pub fn parse_compound_statement(
        &mut self,
        mut meta: Span,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
        terminator: &mut Option<usize>,
    ) -> Result<Span> {
        loop {
            if let Some(Token {
                meta: brace_meta, ..
            }) = self.bump_if(parser, TokenValue::RightBrace)
            {
                meta.subsume(brace_meta);
                break;
            }

            let stmt = self.parse_statement(parser, ctx, body, terminator)?;

            if let Some(stmt_meta) = stmt {
                meta.subsume(stmt_meta);
            }
        }

        if let Some(idx) = *terminator {
            body.cull(idx..)
        }

        Ok(meta)
    }

    pub fn parse_function_args(
        &mut self,
        parser: &mut Parser,
        context: &mut Context,
        body: &mut Block,
    ) -> Result<()> {
        if self.bump_if(parser, TokenValue::Void).is_some() {
            return Ok(());
        }

        loop {
            if self.peek_type_name(parser) || self.peek_parameter_qualifier(parser) {
                let qualifier = self.parse_parameter_qualifier(parser);
                let mut ty = self.parse_type_non_void(parser)?.0;

                match self.expect_peek(parser)?.value {
                    TokenValue::Comma => {
                        self.bump(parser)?;
                        context.add_function_arg(parser, body, None, ty, qualifier);
                        continue;
                    }
                    TokenValue::Identifier(_) => {
                        let mut name = self.expect_ident(parser)?;
                        self.parse_array_specifier(parser, &mut name.1, &mut ty)?;

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
