use crate::{
    front::glsl::{
        ast::{FunctionCall, FunctionCallKind, HirExpr, HirExprKind},
        context::{Context, StmtContext},
        error::{ErrorKind, ExpectedToken},
        parser::ParsingContext,
        token::{Token, TokenValue},
        Error, Parser, Result, Span,
    },
    ArraySize, BinaryOperator, Block, Constant, ConstantInner, Handle, ScalarValue, Type,
    TypeInner, UnaryOperator,
};

impl<'source> ParsingContext<'source> {
    pub fn parse_primary(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
    ) -> Result<Handle<HirExpr>> {
        let mut token = self.bump(parser)?;

        let (width, value) = match token.value {
            TokenValue::IntConstant(int) => (
                (int.width / 8) as u8,
                if int.signed {
                    ScalarValue::Sint(int.value as i64)
                } else {
                    ScalarValue::Uint(int.value)
                },
            ),
            TokenValue::FloatConstant(float) => (
                (float.width / 8) as u8,
                ScalarValue::Float(float.value as f64),
            ),
            TokenValue::BoolConstant(value) => (1, ScalarValue::Bool(value)),
            TokenValue::LeftParen => {
                let expr = self.parse_expression(parser, ctx, stmt, body)?;
                let meta = self.expect(parser, TokenValue::RightParen)?.meta;

                token.meta.subsume(meta);

                return Ok(expr);
            }
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidToken(
                        token.value,
                        vec![
                            TokenValue::LeftParen.into(),
                            ExpectedToken::IntLiteral,
                            ExpectedToken::FloatLiteral,
                            ExpectedToken::BoolLiteral,
                        ],
                    ),
                    meta: token.meta,
                });
            }
        };

        let handle = parser.module.constants.fetch_or_append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar { width, value },
            },
            token.meta,
        );

        Ok(stmt.hir_exprs.append(
            HirExpr {
                kind: HirExprKind::Constant(handle),
                meta: token.meta,
            },
            Default::default(),
        ))
    }

    pub fn parse_function_call_args(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
        meta: &mut Span,
    ) -> Result<Vec<Handle<HirExpr>>> {
        let mut args = Vec::new();
        if let Some(token) = self.bump_if(parser, TokenValue::RightParen) {
            meta.subsume(token.meta);
        } else {
            loop {
                args.push(self.parse_assignment(parser, ctx, stmt, body)?);

                let token = self.bump(parser)?;
                match token.value {
                    TokenValue::Comma => {}
                    TokenValue::RightParen => {
                        meta.subsume(token.meta);
                        break;
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::InvalidToken(
                                token.value,
                                vec![TokenValue::Comma.into(), TokenValue::RightParen.into()],
                            ),
                            meta: token.meta,
                        });
                    }
                }
            }
        }

        Ok(args)
    }

    pub fn parse_postfix(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
    ) -> Result<Handle<HirExpr>> {
        let mut base = if self.peek_type_name(parser) {
            let (mut handle, mut meta) = self.parse_type_non_void(parser)?;

            self.expect(parser, TokenValue::LeftParen)?;
            let args = self.parse_function_call_args(parser, ctx, stmt, body, &mut meta)?;

            if let TypeInner::Array {
                size: ArraySize::Dynamic,
                stride,
                base,
            } = parser.module.types[handle].inner
            {
                let span = parser.module.types.get_span(handle);

                let constant = parser.module.constants.fetch_or_append(
                    Constant {
                        name: None,
                        specialization: None,
                        inner: ConstantInner::Scalar {
                            width: 4,
                            value: ScalarValue::Uint(args.len() as u64),
                        },
                    },
                    Span::default(),
                );
                handle = parser.module.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Array {
                            stride,
                            base,
                            size: ArraySize::Constant(constant),
                        },
                    },
                    span,
                )
            }

            stmt.hir_exprs.append(
                HirExpr {
                    kind: HirExprKind::Call(FunctionCall {
                        kind: FunctionCallKind::TypeConstructor(handle),
                        args,
                    }),
                    meta,
                },
                Default::default(),
            )
        } else if let TokenValue::Identifier(_) = self.expect_peek(parser)?.value {
            let (name, mut meta) = self.expect_ident(parser)?;

            let expr = if self.bump_if(parser, TokenValue::LeftParen).is_some() {
                let args = self.parse_function_call_args(parser, ctx, stmt, body, &mut meta)?;

                let kind = match parser.lookup_type.get(&name) {
                    Some(ty) => FunctionCallKind::TypeConstructor(*ty),
                    None => FunctionCallKind::Function(name),
                };

                HirExpr {
                    kind: HirExprKind::Call(FunctionCall { kind, args }),
                    meta,
                }
            } else {
                let var = match parser.lookup_variable(ctx, body, &name, meta) {
                    Some(var) => var,
                    None => {
                        return Err(Error {
                            kind: ErrorKind::UnknownVariable(name),
                            meta,
                        })
                    }
                };

                HirExpr {
                    kind: HirExprKind::Variable(var),
                    meta,
                }
            };

            stmt.hir_exprs.append(expr, Default::default())
        } else {
            self.parse_primary(parser, ctx, stmt, body)?
        };

        while let TokenValue::LeftBracket
        | TokenValue::Dot
        | TokenValue::Increment
        | TokenValue::Decrement = self.expect_peek(parser)?.value
        {
            let Token { value, mut meta } = self.bump(parser)?;

            match value {
                TokenValue::LeftBracket => {
                    let index = self.parse_expression(parser, ctx, stmt, body)?;
                    let end_meta = self.expect(parser, TokenValue::RightBracket)?.meta;

                    meta.subsume(end_meta);
                    base = stmt.hir_exprs.append(
                        HirExpr {
                            kind: HirExprKind::Access { base, index },
                            meta,
                        },
                        Default::default(),
                    )
                }
                TokenValue::Dot => {
                    let (field, end_meta) = self.expect_ident(parser)?;

                    meta.subsume(end_meta);
                    base = stmt.hir_exprs.append(
                        HirExpr {
                            kind: HirExprKind::Select { base, field },
                            meta,
                        },
                        Default::default(),
                    )
                }
                TokenValue::Increment | TokenValue::Decrement => {
                    base = stmt.hir_exprs.append(
                        HirExpr {
                            kind: HirExprKind::PrePostfix {
                                op: match value {
                                    TokenValue::Increment => crate::BinaryOperator::Add,
                                    _ => crate::BinaryOperator::Subtract,
                                },
                                postfix: true,
                                expr: base,
                            },
                            meta,
                        },
                        Default::default(),
                    )
                }
                _ => unreachable!(),
            }
        }

        Ok(base)
    }

    pub fn parse_unary(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
    ) -> Result<Handle<HirExpr>> {
        Ok(match self.expect_peek(parser)?.value {
            TokenValue::Plus | TokenValue::Dash | TokenValue::Bang | TokenValue::Tilde => {
                let Token { value, mut meta } = self.bump(parser)?;

                let expr = self.parse_unary(parser, ctx, stmt, body)?;
                let end_meta = stmt.hir_exprs[expr].meta;

                let kind = match value {
                    TokenValue::Dash => HirExprKind::Unary {
                        op: UnaryOperator::Negate,
                        expr,
                    },
                    TokenValue::Bang | TokenValue::Tilde => HirExprKind::Unary {
                        op: UnaryOperator::Not,
                        expr,
                    },
                    _ => return Ok(expr),
                };

                meta.subsume(end_meta);
                stmt.hir_exprs
                    .append(HirExpr { kind, meta }, Default::default())
            }
            TokenValue::Increment | TokenValue::Decrement => {
                let Token { value, meta } = self.bump(parser)?;

                let expr = self.parse_unary(parser, ctx, stmt, body)?;

                stmt.hir_exprs.append(
                    HirExpr {
                        kind: HirExprKind::PrePostfix {
                            op: match value {
                                TokenValue::Increment => crate::BinaryOperator::Add,
                                _ => crate::BinaryOperator::Subtract,
                            },
                            postfix: false,
                            expr,
                        },
                        meta,
                    },
                    Default::default(),
                )
            }
            _ => self.parse_postfix(parser, ctx, stmt, body)?,
        })
    }

    pub fn parse_binary(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
        passtrough: Option<Handle<HirExpr>>,
        min_bp: u8,
    ) -> Result<Handle<HirExpr>> {
        let mut left = passtrough
            .ok_or(ErrorKind::EndOfFile /* Dummy error */)
            .or_else(|_| self.parse_unary(parser, ctx, stmt, body))?;
        let mut meta = stmt.hir_exprs[left].meta;

        while let Some((l_bp, r_bp)) = binding_power(&self.expect_peek(parser)?.value) {
            if l_bp < min_bp {
                break;
            }

            let Token { value, .. } = self.bump(parser)?;

            let right = self.parse_binary(parser, ctx, stmt, body, None, r_bp)?;
            let end_meta = stmt.hir_exprs[right].meta;

            meta.subsume(end_meta);
            left = stmt.hir_exprs.append(
                HirExpr {
                    kind: HirExprKind::Binary {
                        left,
                        op: match value {
                            TokenValue::LogicalOr => BinaryOperator::LogicalOr,
                            TokenValue::LogicalXor => BinaryOperator::NotEqual,
                            TokenValue::LogicalAnd => BinaryOperator::LogicalAnd,
                            TokenValue::VerticalBar => BinaryOperator::InclusiveOr,
                            TokenValue::Caret => BinaryOperator::ExclusiveOr,
                            TokenValue::Ampersand => BinaryOperator::And,
                            TokenValue::Equal => BinaryOperator::Equal,
                            TokenValue::NotEqual => BinaryOperator::NotEqual,
                            TokenValue::GreaterEqual => BinaryOperator::GreaterEqual,
                            TokenValue::LessEqual => BinaryOperator::LessEqual,
                            TokenValue::LeftAngle => BinaryOperator::Less,
                            TokenValue::RightAngle => BinaryOperator::Greater,
                            TokenValue::LeftShift => BinaryOperator::ShiftLeft,
                            TokenValue::RightShift => BinaryOperator::ShiftRight,
                            TokenValue::Plus => BinaryOperator::Add,
                            TokenValue::Dash => BinaryOperator::Subtract,
                            TokenValue::Star => BinaryOperator::Multiply,
                            TokenValue::Slash => BinaryOperator::Divide,
                            TokenValue::Percent => BinaryOperator::Modulo,
                            _ => unreachable!(),
                        },
                        right,
                    },
                    meta,
                },
                Default::default(),
            )
        }

        Ok(left)
    }

    pub fn parse_conditional(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
        passtrough: Option<Handle<HirExpr>>,
    ) -> Result<Handle<HirExpr>> {
        let mut condition = self.parse_binary(parser, ctx, stmt, body, passtrough, 0)?;
        let mut meta = stmt.hir_exprs[condition].meta;

        if self.bump_if(parser, TokenValue::Question).is_some() {
            let accept = self.parse_expression(parser, ctx, stmt, body)?;
            self.expect(parser, TokenValue::Colon)?;
            let reject = self.parse_assignment(parser, ctx, stmt, body)?;
            let end_meta = stmt.hir_exprs[reject].meta;

            meta.subsume(end_meta);
            condition = stmt.hir_exprs.append(
                HirExpr {
                    kind: HirExprKind::Conditional {
                        condition,
                        accept,
                        reject,
                    },
                    meta,
                },
                Default::default(),
            )
        }

        Ok(condition)
    }

    pub fn parse_assignment(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
    ) -> Result<Handle<HirExpr>> {
        let tgt = self.parse_unary(parser, ctx, stmt, body)?;
        let mut meta = stmt.hir_exprs[tgt].meta;

        Ok(match self.expect_peek(parser)?.value {
            TokenValue::Assign => {
                self.bump(parser)?;
                let value = self.parse_assignment(parser, ctx, stmt, body)?;
                let end_meta = stmt.hir_exprs[value].meta;

                meta.subsume(end_meta);
                stmt.hir_exprs.append(
                    HirExpr {
                        kind: HirExprKind::Assign { tgt, value },
                        meta,
                    },
                    Default::default(),
                )
            }
            TokenValue::OrAssign
            | TokenValue::AndAssign
            | TokenValue::AddAssign
            | TokenValue::DivAssign
            | TokenValue::ModAssign
            | TokenValue::SubAssign
            | TokenValue::MulAssign
            | TokenValue::LeftShiftAssign
            | TokenValue::RightShiftAssign
            | TokenValue::XorAssign => {
                let token = self.bump(parser)?;
                let right = self.parse_assignment(parser, ctx, stmt, body)?;
                let end_meta = stmt.hir_exprs[right].meta;

                meta.subsume(end_meta);
                let value = stmt.hir_exprs.append(
                    HirExpr {
                        meta,
                        kind: HirExprKind::Binary {
                            left: tgt,
                            op: match token.value {
                                TokenValue::OrAssign => BinaryOperator::InclusiveOr,
                                TokenValue::AndAssign => BinaryOperator::And,
                                TokenValue::AddAssign => BinaryOperator::Add,
                                TokenValue::DivAssign => BinaryOperator::Divide,
                                TokenValue::ModAssign => BinaryOperator::Modulo,
                                TokenValue::SubAssign => BinaryOperator::Subtract,
                                TokenValue::MulAssign => BinaryOperator::Multiply,
                                TokenValue::LeftShiftAssign => BinaryOperator::ShiftLeft,
                                TokenValue::RightShiftAssign => BinaryOperator::ShiftRight,
                                TokenValue::XorAssign => BinaryOperator::ExclusiveOr,
                                _ => unreachable!(),
                            },
                            right,
                        },
                    },
                    Default::default(),
                );

                stmt.hir_exprs.append(
                    HirExpr {
                        kind: HirExprKind::Assign { tgt, value },
                        meta,
                    },
                    Default::default(),
                )
            }
            _ => self.parse_conditional(parser, ctx, stmt, body, Some(tgt))?,
        })
    }

    pub fn parse_expression(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        stmt: &mut StmtContext,
        body: &mut Block,
    ) -> Result<Handle<HirExpr>> {
        let mut expr = self.parse_assignment(parser, ctx, stmt, body)?;

        while let TokenValue::Comma = self.expect_peek(parser)?.value {
            self.bump(parser)?;
            expr = self.parse_assignment(parser, ctx, stmt, body)?;
        }

        Ok(expr)
    }
}

fn binding_power(value: &TokenValue) -> Option<(u8, u8)> {
    Some(match *value {
        TokenValue::LogicalOr => (1, 2),
        TokenValue::LogicalXor => (3, 4),
        TokenValue::LogicalAnd => (5, 6),
        TokenValue::VerticalBar => (7, 8),
        TokenValue::Caret => (9, 10),
        TokenValue::Ampersand => (11, 12),
        TokenValue::Equal | TokenValue::NotEqual => (13, 14),
        TokenValue::GreaterEqual
        | TokenValue::LessEqual
        | TokenValue::LeftAngle
        | TokenValue::RightAngle => (15, 16),
        TokenValue::LeftShift | TokenValue::RightShift => (17, 18),
        TokenValue::Plus | TokenValue::Dash => (19, 20),
        TokenValue::Star | TokenValue::Slash | TokenValue::Percent => (21, 22),
        _ => return None,
    })
}
