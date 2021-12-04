use crate::{
    front::glsl::{
        ast::{
            GlobalLookup, GlobalLookupKind, Precision, StorageQualifier, StructLayout,
            TypeQualifier,
        },
        context::{Context, ExprPos},
        error::ExpectedToken,
        offset,
        token::{Token, TokenValue},
        types::scalar_components,
        variables::{GlobalOrConstant, VarDeclaration},
        Error, ErrorKind, Parser, Span,
    },
    Block, Expression, FunctionResult, Handle, ScalarKind, Statement, StorageClass, StructMember,
    Type, TypeInner,
};

use super::{DeclarationContext, ParsingContext, Result};

impl<'source> ParsingContext<'source> {
    pub fn parse_external_declaration(
        &mut self,
        parser: &mut Parser,
        global_ctx: &mut Context,
        global_body: &mut Block,
    ) -> Result<()> {
        if self
            .parse_declaration(parser, global_ctx, global_body, true)?
            .is_none()
        {
            let token = self.bump(parser)?;
            match token.value {
                TokenValue::Semicolon if parser.meta.version == 460 => Ok(()),
                _ => {
                    let expected = match parser.meta.version {
                        460 => vec![TokenValue::Semicolon.into(), ExpectedToken::Eof],
                        _ => vec![ExpectedToken::Eof],
                    };
                    Err(Error {
                        kind: ErrorKind::InvalidToken(token.value, expected),
                        meta: token.meta,
                    })
                }
            }
        } else {
            Ok(())
        }
    }

    pub fn parse_initializer(
        &mut self,
        parser: &mut Parser,
        ty: Handle<Type>,
        ctx: &mut Context,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Span)> {
        // initializer:
        //     assignment_expression
        //     LEFT_BRACE initializer_list RIGHT_BRACE
        //     LEFT_BRACE initializer_list COMMA RIGHT_BRACE
        //
        // initializer_list:
        //     initializer
        //     initializer_list COMMA initializer
        if let Some(Token { mut meta, .. }) = self.bump_if(parser, TokenValue::LeftBrace) {
            // initializer_list
            let mut components = Vec::new();
            loop {
                // TODO: Change type
                components.push(self.parse_initializer(parser, ty, ctx, body)?.0);

                let token = self.bump(parser)?;
                match token.value {
                    TokenValue::Comma => {
                        if let Some(Token { meta: end_meta, .. }) =
                            self.bump_if(parser, TokenValue::RightBrace)
                        {
                            meta.subsume(end_meta);
                            break;
                        }
                    }
                    TokenValue::RightBrace => {
                        meta.subsume(token.meta);
                        break;
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::InvalidToken(
                                token.value,
                                vec![TokenValue::Comma.into(), TokenValue::RightBrace.into()],
                            ),
                            meta: token.meta,
                        })
                    }
                }
            }

            Ok((
                ctx.add_expression(Expression::Compose { ty, components }, meta, body),
                meta,
            ))
        } else {
            let mut stmt = ctx.stmt_ctx();
            let expr = self.parse_assignment(parser, ctx, &mut stmt, body)?;
            let (mut init, init_meta) = ctx.lower_expect(stmt, parser, expr, ExprPos::Rhs, body)?;

            let scalar_components = scalar_components(&parser.module.types[ty].inner);
            if let Some((kind, width)) = scalar_components {
                ctx.implicit_conversion(parser, &mut init, init_meta, kind, width)?;
            }

            Ok((init, init_meta))
        }
    }

    // Note: caller preparsed the type and qualifiers
    // Note: caller skips this if the fallthrough token is not expected to be consumed here so this
    // produced Error::InvalidToken if it isn't consumed
    pub fn parse_init_declarator_list(
        &mut self,
        parser: &mut Parser,
        ty: Handle<Type>,
        mut fallthrough: Option<Token>,
        ctx: &mut DeclarationContext,
    ) -> Result<()> {
        // init_declarator_list:
        //     single_declaration
        //     init_declarator_list COMMA IDENTIFIER
        //     init_declarator_list COMMA IDENTIFIER array_specifier
        //     init_declarator_list COMMA IDENTIFIER array_specifier EQUAL initializer
        //     init_declarator_list COMMA IDENTIFIER EQUAL initializer
        //
        // single_declaration:
        //     fully_specified_type
        //     fully_specified_type IDENTIFIER
        //     fully_specified_type IDENTIFIER array_specifier
        //     fully_specified_type IDENTIFIER array_specifier EQUAL initializer
        //     fully_specified_type IDENTIFIER EQUAL initializer

        // Consume any leading comma, e.g. this is valid: `float, a=1;`
        if fallthrough
            .as_ref()
            .or_else(|| self.peek(parser))
            .filter(|t| t.value == TokenValue::Comma)
            .is_some()
        {
            fallthrough.take().or_else(|| self.next(parser));
        }

        loop {
            let token = fallthrough
                .take()
                .ok_or(ErrorKind::EndOfFile)
                .or_else(|_| self.bump(parser))?;
            let name = match token.value {
                TokenValue::Semicolon => break,
                TokenValue::Identifier(name) => name,
                _ => {
                    return Err(Error {
                        kind: ErrorKind::InvalidToken(
                            token.value,
                            vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                        ),
                        meta: token.meta,
                    })
                }
            };
            let mut meta = token.meta;

            // array_specifier
            // array_specifier EQUAL initializer
            // EQUAL initializer

            // parse an array specifier if it exists
            // NOTE: unlike other parse methods this one doesn't expect an array specifier and
            // returns Ok(None) rather than an error if there is not one
            let array_specifier = self.parse_array_specifier(parser)?;
            let ty = parser.maybe_array(ty, meta, array_specifier);

            let init = self
                .bump_if(parser, TokenValue::Assign)
                .map::<Result<_>, _>(|_| {
                    let (mut expr, init_meta) =
                        self.parse_initializer(parser, ty, ctx.ctx, ctx.body)?;

                    let scalar_components = scalar_components(&parser.module.types[ty].inner);
                    if let Some((kind, width)) = scalar_components {
                        ctx.ctx
                            .implicit_conversion(parser, &mut expr, init_meta, kind, width)?;
                    }

                    meta.subsume(init_meta);

                    Ok((expr, init_meta))
                })
                .transpose()?;

            // TODO: Should we try to make constants here?
            // This is mostly a hack because we don't yet support adding
            // bodies to entry points for variable initialization
            let maybe_constant =
                init.and_then(|(root, meta)| parser.solve_constant(ctx.ctx, root, meta).ok());

            let pointer = ctx.add_var(parser, ty, name, maybe_constant, meta)?;

            if let Some((value, _)) = init.filter(|_| maybe_constant.is_none()) {
                ctx.flush_expressions();
                ctx.body.push(Statement::Store { pointer, value }, meta);
            }

            let token = self.bump(parser)?;
            match token.value {
                TokenValue::Semicolon => break,
                TokenValue::Comma => {}
                _ => {
                    return Err(Error {
                        kind: ErrorKind::InvalidToken(
                            token.value,
                            vec![TokenValue::Comma.into(), TokenValue::Semicolon.into()],
                        ),
                        meta: token.meta,
                    })
                }
            }
        }

        Ok(())
    }

    /// `external` whether or not we are in a global or local context
    pub fn parse_declaration(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
        external: bool,
    ) -> Result<Option<Span>> {
        //declaration:
        //    function_prototype  SEMICOLON
        //
        //    init_declarator_list SEMICOLON
        //    PRECISION precision_qualifier type_specifier SEMICOLON
        //
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE SEMICOLON
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE IDENTIFIER SEMICOLON
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE IDENTIFIER array_specifier SEMICOLON
        //    type_qualifier SEMICOLON type_qualifier IDENTIFIER SEMICOLON
        //    type_qualifier IDENTIFIER identifier_list SEMICOLON

        if self.peek_type_qualifier(parser) || self.peek_type_name(parser) {
            let qualifiers = self.parse_type_qualifiers(parser)?;

            if self.peek_type_name(parser) {
                // This branch handles variables and function prototypes and if
                // external is true also function definitions
                let (ty, mut meta) = self.parse_type(parser)?;

                let token = self.bump(parser)?;
                let token_fallthrough = match token.value {
                    TokenValue::Identifier(name) => match self.expect_peek(parser)?.value {
                        TokenValue::LeftParen => {
                            // This branch handles function definition and prototypes
                            self.bump(parser)?;

                            let result = ty.map(|ty| FunctionResult { ty, binding: None });
                            let mut body = Block::new();

                            let mut context = Context::new(parser, &mut body);

                            self.parse_function_args(parser, &mut context, &mut body)?;

                            let end_meta = self.expect(parser, TokenValue::RightParen)?.meta;
                            meta.subsume(end_meta);

                            let token = self.bump(parser)?;
                            return match token.value {
                                TokenValue::Semicolon => {
                                    // This branch handles function prototypes
                                    parser.add_prototype(context, name, result, meta);

                                    Ok(Some(meta))
                                }
                                TokenValue::LeftBrace if external => {
                                    // This branch handles function definitions
                                    // as you can see by the guard this branch
                                    // only happens if external is also true

                                    // parse the body
                                    self.parse_compound_statement(
                                        token.meta,
                                        parser,
                                        &mut context,
                                        &mut body,
                                        &mut None,
                                    )?;

                                    parser.add_function(context, name, result, body, meta);

                                    Ok(Some(meta))
                                }
                                _ if external => Err(Error {
                                    kind: ErrorKind::InvalidToken(
                                        token.value,
                                        vec![
                                            TokenValue::LeftBrace.into(),
                                            TokenValue::Semicolon.into(),
                                        ],
                                    ),
                                    meta: token.meta,
                                }),
                                _ => Err(Error {
                                    kind: ErrorKind::InvalidToken(
                                        token.value,
                                        vec![TokenValue::Semicolon.into()],
                                    ),
                                    meta: token.meta,
                                }),
                            };
                        }
                        // Pass the token to the init_declator_list parser
                        _ => Token {
                            value: TokenValue::Identifier(name),
                            meta: token.meta,
                        },
                    },
                    // Pass the token to the init_declator_list parser
                    _ => token,
                };

                // If program execution has reached here then this will be a
                // init_declarator_list
                // token_falltrough will have a token that was already bumped
                if let Some(ty) = ty {
                    let mut ctx = DeclarationContext {
                        qualifiers,
                        external,
                        ctx,
                        body,
                    };

                    self.parse_init_declarator_list(parser, ty, Some(token_fallthrough), &mut ctx)?;
                } else {
                    parser.errors.push(Error {
                        kind: ErrorKind::SemanticError("Declaration cannot have void type".into()),
                        meta,
                    })
                }

                Ok(Some(meta))
            } else {
                // This branch handles struct definitions and modifiers like
                // ```glsl
                // layout(early_fragment_tests);
                // ```
                let token = self.bump(parser)?;
                match token.value {
                    TokenValue::Identifier(ty_name) => {
                        if self.bump_if(parser, TokenValue::LeftBrace).is_some() {
                            self.parse_block_declaration(
                                parser,
                                ctx,
                                body,
                                &qualifiers,
                                ty_name,
                                token.meta,
                            )
                            .map(Some)
                        } else {
                            //TODO: declaration
                            // type_qualifier IDENTIFIER SEMICOLON
                            // type_qualifier IDENTIFIER identifier_list SEMICOLON
                            Err(Error {
                                kind: ErrorKind::NotImplemented("variable qualifier"),
                                meta: token.meta,
                            })
                        }
                    }
                    TokenValue::Semicolon => {
                        let mut meta_all = token.meta;
                        for &(ref qualifier, meta) in qualifiers.iter() {
                            meta_all.subsume(meta);
                            match *qualifier {
                                TypeQualifier::WorkGroupSize(i, value) => {
                                    parser.meta.workgroup_size[i] = value
                                }
                                TypeQualifier::EarlyFragmentTests => {
                                    parser.meta.early_fragment_tests = true;
                                }
                                TypeQualifier::StorageQualifier(_) => {
                                    // TODO: Maybe add some checks here
                                    // This is needed because of cases like
                                    // layout(early_fragment_tests) in;
                                }
                                _ => {
                                    parser.errors.push(Error {
                                        kind: ErrorKind::SemanticError(
                                            "Qualifier not supported as standalone".into(),
                                        ),
                                        meta,
                                    });
                                }
                            }
                        }

                        Ok(Some(meta_all))
                    }
                    _ => Err(Error {
                        kind: ErrorKind::InvalidToken(
                            token.value,
                            vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                        ),
                        meta: token.meta,
                    }),
                }
            }
        } else {
            match self.peek(parser).map(|t| &t.value) {
                Some(&TokenValue::Precision) => {
                    // PRECISION precision_qualifier type_specifier SEMICOLON
                    self.bump(parser)?;

                    let token = self.bump(parser)?;
                    let _ = match token.value {
                        TokenValue::PrecisionQualifier(p) => p,
                        _ => {
                            return Err(Error {
                                kind: ErrorKind::InvalidToken(
                                    token.value,
                                    vec![
                                        TokenValue::PrecisionQualifier(Precision::High).into(),
                                        TokenValue::PrecisionQualifier(Precision::Medium).into(),
                                        TokenValue::PrecisionQualifier(Precision::Low).into(),
                                    ],
                                ),
                                meta: token.meta,
                            })
                        }
                    };

                    let (ty, meta) = self.parse_type_non_void(parser)?;

                    match parser.module.types[ty].inner {
                        TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            ..
                        }
                        | TypeInner::Scalar {
                            kind: ScalarKind::Sint,
                            ..
                        } => {}
                        _ => parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Precision statement can only work on floats and ints".into(),
                            ),
                            meta,
                        }),
                    }

                    self.expect(parser, TokenValue::Semicolon)?;

                    Ok(Some(meta))
                }
                _ => Ok(None),
            }
        }
    }

    pub fn parse_block_declaration(
        &mut self,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
        qualifiers: &[(TypeQualifier, Span)],
        ty_name: String,
        meta: Span,
    ) -> Result<Span> {
        let mut storage = None;
        let mut layout = None;

        for &(ref qualifier, _) in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(StorageQualifier::StorageClass(c)) => {
                    storage = Some(c)
                }
                TypeQualifier::Layout(l) => layout = Some(l),
                _ => continue,
            }
        }

        let layout = match (layout, storage) {
            (Some(layout), _) => layout,
            (None, Some(StorageClass::Storage { .. })) => StructLayout::Std430,
            _ => StructLayout::Std140,
        };

        let mut members = Vec::new();
        let span = self.parse_struct_declaration_list(parser, &mut members, layout)?;
        self.expect(parser, TokenValue::RightBrace)?;

        let mut ty = parser.module.types.insert(
            Type {
                name: Some(ty_name),
                inner: TypeInner::Struct {
                    members: members.clone(),
                    span,
                },
            },
            Default::default(),
        );

        let token = self.bump(parser)?;
        let name = match token.value {
            TokenValue::Semicolon => None,
            TokenValue::Identifier(name) => {
                let array_specifier = self.parse_array_specifier(parser)?;
                ty = parser.maybe_array(ty, token.meta, array_specifier);

                self.expect(parser, TokenValue::Semicolon)?;

                Some(name)
            }
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidToken(
                        token.value,
                        vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                    ),
                    meta: token.meta,
                })
            }
        };

        let global = parser.add_global_var(
            ctx,
            body,
            VarDeclaration {
                qualifiers,
                ty,
                name,
                init: None,
                meta,
            },
        )?;

        for (i, k, ty) in members.into_iter().enumerate().filter_map(|(i, m)| {
            let ty = m.ty;
            m.name.map(|s| (i as u32, s, ty))
        }) {
            let lookup = GlobalLookup {
                kind: match global {
                    GlobalOrConstant::Global(handle) => GlobalLookupKind::BlockSelect(handle, i),
                    GlobalOrConstant::Constant(handle) => GlobalLookupKind::Constant(handle, ty),
                },
                entry_arg: None,
                mutable: true,
            };
            ctx.add_global(parser, &k, lookup, body);

            parser.global_variables.push((k, lookup));
        }

        Ok(meta)
    }

    // TODO: Accept layout arguments
    pub fn parse_struct_declaration_list(
        &mut self,
        parser: &mut Parser,
        members: &mut Vec<StructMember>,
        layout: StructLayout,
    ) -> Result<u32> {
        let mut span = 0;
        let mut align = 0;

        loop {
            // TODO: type_qualifier

            let (ty, mut meta) = self.parse_type_non_void(parser)?;
            let (name, end_meta) = self.expect_ident(parser)?;

            meta.subsume(end_meta);

            let array_specifier = self.parse_array_specifier(parser)?;
            let ty = parser.maybe_array(ty, meta, array_specifier);

            self.expect(parser, TokenValue::Semicolon)?;

            let info = offset::calculate_offset(
                ty,
                meta,
                layout,
                &mut parser.module.types,
                &parser.module.constants,
                &mut parser.errors,
            );

            span = crate::front::align_up(span, info.align);
            align = align.max(info.align);

            members.push(StructMember {
                name: Some(name),
                ty: info.ty,
                binding: None,
                offset: span,
            });

            span += info.span;

            if let TokenValue::RightBrace = self.expect_peek(parser)?.value {
                break;
            }
        }

        span = crate::front::align_up(span, align);

        Ok(span)
    }
}
