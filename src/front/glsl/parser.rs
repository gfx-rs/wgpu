use super::{
    ast::{
        Context, FunctionCall, FunctionCallKind, FunctionSignature, GlobalLookup, GlobalLookupKind,
        HirExpr, HirExprKind, ParameterQualifier, Profile, StorageQualifier, StructLayout,
        TypeQualifier,
    },
    error::ErrorKind,
    lex::Lexer,
    token::{SourceMetadata, Token, TokenValue},
    variables::{GlobalOrConstant, VarDeclaration},
    Program,
};
use crate::{
    arena::Handle, front::glsl::error::ExpectedToken, Arena, ArraySize, BinaryOperator, Block,
    Constant, ConstantInner, Expression, Function, FunctionResult, ResourceBinding, ScalarValue,
    Statement, StorageClass, StructMember, SwitchCase, Type, TypeInner, UnaryOperator,
};
use core::convert::TryFrom;
use std::{iter::Peekable, mem};

type Result<T> = std::result::Result<T, ErrorKind>;

pub struct Parser<'source, 'program, 'options> {
    program: &'program mut Program<'options>,
    lexer: Peekable<Lexer<'source>>,
}

impl<'source, 'program, 'options> Parser<'source, 'program, 'options> {
    pub fn new(program: &'program mut Program<'options>, lexer: Lexer<'source>) -> Self {
        Parser {
            program,
            lexer: lexer.peekable(),
        }
    }

    fn expect_ident(&mut self) -> Result<(String, SourceMetadata)> {
        let token = self.bump()?;

        match token.value {
            TokenValue::Identifier(name) => Ok((name, token.meta)),
            _ => Err(ErrorKind::InvalidToken(
                token,
                vec![ExpectedToken::Identifier],
            )),
        }
    }

    fn expect(&mut self, value: TokenValue) -> Result<Token> {
        let token = self.bump()?;

        if token.value != value {
            Err(ErrorKind::InvalidToken(token, vec![value.into()]))
        } else {
            Ok(token)
        }
    }

    fn bump(&mut self) -> Result<Token> {
        self.lexer.next().ok_or(ErrorKind::EndOfFile)
    }

    /// Returns None on the end of the file rather than an error like other methods
    fn bump_if(&mut self, value: TokenValue) -> Option<Token> {
        if self.lexer.peek().filter(|t| t.value == value).is_some() {
            self.bump().ok()
        } else {
            None
        }
    }

    fn expect_peek(&mut self) -> Result<&Token> {
        self.lexer.peek().ok_or(ErrorKind::EndOfFile)
    }

    pub fn parse(&mut self) -> Result<()> {
        self.parse_version()?;

        while self.lexer.peek().is_some() {
            self.parse_external_declaration()?;
        }

        self.program.add_entry_points();

        Ok(())
    }

    fn parse_version(&mut self) -> Result<()> {
        self.expect(TokenValue::Version)?;

        let version = self.bump()?;
        match version.value {
            TokenValue::IntConstant(i) => match i.value {
                440 | 450 | 460 => self.program.version = i.value as u16,
                _ => return Err(ErrorKind::InvalidVersion(version.meta, i.value)),
            },
            _ => {
                return Err(ErrorKind::InvalidToken(
                    version,
                    vec![ExpectedToken::IntLiteral],
                ))
            }
        }

        let profile = self.lexer.peek();
        self.program.profile = match profile {
            Some(&Token {
                value: TokenValue::Identifier(_),
                ..
            }) => {
                let (name, meta) = self.expect_ident()?;

                match name.as_str() {
                    "core" => Profile::Core,
                    _ => return Err(ErrorKind::InvalidProfile(meta, name)),
                }
            }
            _ => Profile::Core,
        };

        Ok(())
    }

    /// Parses an optional array_specifier returning `Ok(None)` if there is no
    /// LeftBracket
    fn parse_array_specifier(&mut self) -> Result<Option<ArraySize>> {
        if self.bump_if(TokenValue::LeftBracket).is_some() {
            if self.bump_if(TokenValue::RightBracket).is_some() {
                return Ok(Some(ArraySize::Dynamic));
            }

            let (constant, _) = self.parse_constant_expression()?;
            self.expect(TokenValue::RightBracket)?;
            Ok(Some(ArraySize::Constant(constant)))
        } else {
            Ok(None)
        }
    }

    fn parse_type(&mut self) -> Result<(Option<Handle<Type>>, SourceMetadata)> {
        let token = self.bump()?;
        let handle = match token.value {
            TokenValue::Void => None,
            TokenValue::TypeName(ty) => Some(self.program.module.types.fetch_or_append(ty)),
            TokenValue::Struct => {
                let ty_name = self.expect_ident()?.0;
                self.expect(TokenValue::LeftBrace)?;
                let mut members = Vec::new();
                let span = self.parse_struct_declaration_list(&mut members)?;
                self.expect(TokenValue::RightBrace)?;

                let ty = self.program.module.types.append(Type {
                    name: Some(ty_name.clone()),
                    inner: TypeInner::Struct {
                        top_level: false,
                        members,
                        span,
                    },
                });
                self.program.lookup_type.insert(ty_name, ty);
                Some(ty)
            }
            TokenValue::Identifier(ident) => match self.program.lookup_type.get(&ident) {
                Some(ty) => Some(*ty),
                None => return Err(ErrorKind::UnknownType(token.meta, ident)),
            },
            _ => {
                return Err(ErrorKind::InvalidToken(
                    token,
                    vec![
                        TokenValue::Void.into(),
                        TokenValue::Struct.into(),
                        ExpectedToken::TypeName,
                    ],
                ))
            }
        };

        let size = self.parse_array_specifier()?;
        Ok((handle.map(|ty| self.maybe_array(ty, size)), token.meta))
    }

    fn parse_type_non_void(&mut self) -> Result<(Handle<Type>, SourceMetadata)> {
        let (maybe_ty, meta) = self.parse_type()?;
        let ty =
            maybe_ty.ok_or_else(|| ErrorKind::SemanticError(meta, "Type can't be void".into()))?;

        Ok((ty, meta))
    }

    fn maybe_array(&mut self, base: Handle<Type>, size: Option<ArraySize>) -> Handle<Type> {
        size.map(|size| {
            self.program.module.types.fetch_or_append(Type {
                name: None,
                inner: TypeInner::Array {
                    base,
                    size,
                    stride: self.program.module.types[base]
                        .inner
                        .span(&self.program.module.constants),
                },
            })
        })
        .unwrap_or(base)
    }

    fn peek_type_qualifier(&mut self) -> bool {
        self.lexer.peek().map_or(false, |t| match t.value {
            TokenValue::Interpolation(_)
            | TokenValue::Sampling(_)
            | TokenValue::Const
            | TokenValue::In
            | TokenValue::Out
            | TokenValue::Uniform
            | TokenValue::Buffer
            | TokenValue::Layout => true,
            _ => false,
        })
    }

    fn parse_type_qualifiers(&mut self) -> Result<Vec<(TypeQualifier, SourceMetadata)>> {
        let mut qualifiers = Vec::new();

        while self.peek_type_qualifier() {
            let token = self.bump()?;

            // Handle layout qualifiers outside the match since this can push multiple values
            if token.value == TokenValue::Layout {
                self.parse_layout_qualifier_id_list(&mut qualifiers)?;
                continue;
            }

            qualifiers.push((
                match token.value {
                    TokenValue::Interpolation(i) => TypeQualifier::Interpolation(i),
                    TokenValue::Const => TypeQualifier::StorageQualifier(StorageQualifier::Const),
                    TokenValue::In => TypeQualifier::StorageQualifier(StorageQualifier::Input),
                    TokenValue::Out => TypeQualifier::StorageQualifier(StorageQualifier::Output),
                    TokenValue::Uniform => TypeQualifier::StorageQualifier(
                        StorageQualifier::StorageClass(StorageClass::Uniform),
                    ),
                    TokenValue::Buffer => TypeQualifier::StorageQualifier(
                        StorageQualifier::StorageClass(StorageClass::Storage),
                    ),
                    TokenValue::Sampling(s) => TypeQualifier::Sampling(s),

                    _ => unreachable!(),
                },
                token.meta,
            ))
        }

        Ok(qualifiers)
    }

    fn parse_layout_qualifier_id_list(
        &mut self,
        qualifiers: &mut Vec<(TypeQualifier, SourceMetadata)>,
    ) -> Result<()> {
        // We need both of these to produce a ResourceBinding
        let mut group = None;
        let mut binding = None;

        self.expect(TokenValue::LeftParen)?;
        loop {
            self.parse_layout_qualifier_id(qualifiers, &mut group, &mut binding)?;

            if self.bump_if(TokenValue::Comma).is_some() {
                continue;
            }

            break;
        }
        self.expect(TokenValue::RightParen)?;

        match (group, binding) {
            (Some((group, group_meta)), Some((binding, binding_meta))) => qualifiers.push((
                TypeQualifier::ResourceBinding(ResourceBinding { group, binding }),
                group_meta.union(&binding_meta),
            )),
            // Produce an error if we have one of group or binding but not the other
            (Some((_, meta)), None) => {
                return Err(ErrorKind::SemanticError(
                    meta,
                    "set specified with no binding".into(),
                ))
            }
            (None, Some((_, meta))) => {
                return Err(ErrorKind::SemanticError(
                    meta,
                    "binding specified with no set".into(),
                ))
            }
            (None, None) => (),
        }

        Ok(())
    }

    fn parse_uint_constant(&mut self) -> Result<(u32, SourceMetadata)> {
        let (value, meta) = self.parse_constant_expression()?;

        let int = match self.program.module.constants[value].inner {
            ConstantInner::Scalar {
                value: ScalarValue::Uint(int),
                ..
            } => u32::try_from(int)
                .map_err(|_| ErrorKind::SemanticError(meta, "int constant overflows".into())),
            ConstantInner::Scalar {
                value: ScalarValue::Sint(int),
                ..
            } => u32::try_from(int)
                .map_err(|_| ErrorKind::SemanticError(meta, "int constant overflows".into())),
            _ => Err(ErrorKind::SemanticError(
                meta,
                "Expected a uint constant".into(),
            )),
        }?;

        Ok((int, meta))
    }

    fn parse_layout_qualifier_id(
        &mut self,
        qualifiers: &mut Vec<(TypeQualifier, SourceMetadata)>,
        group: &mut Option<(u32, SourceMetadata)>,
        binding: &mut Option<(u32, SourceMetadata)>,
    ) -> Result<()> {
        // layout_qualifier_id:
        //     IDENTIFIER
        //     IDENTIFIER EQUAL constant_expression
        //     SHARED
        let mut token = self.bump()?;
        match token.value {
            TokenValue::Identifier(name) => {
                if self.bump_if(TokenValue::Assign).is_some() {
                    let (value, end_meta) = self.parse_uint_constant()?;
                    token.meta = token.meta.union(&end_meta);

                    qualifiers.push((
                        match name.as_str() {
                            "location" => TypeQualifier::Location(value),
                            "set" => {
                                *group = Some((value, end_meta));
                                return Ok(());
                            }
                            "binding" => {
                                *binding = Some((value, end_meta));
                                return Ok(());
                            }
                            "local_size_x" => TypeQualifier::WorkGroupSize(0, value),
                            "local_size_y" => TypeQualifier::WorkGroupSize(1, value),
                            "local_size_z" => TypeQualifier::WorkGroupSize(2, value),
                            _ => return Err(ErrorKind::UnknownLayoutQualifier(token.meta, name)),
                        },
                        token.meta,
                    ))
                } else {
                    match name.as_str() {
                        "push_constant" => {
                            qualifiers.push((
                                TypeQualifier::StorageQualifier(StorageQualifier::StorageClass(
                                    StorageClass::PushConstant,
                                )),
                                token.meta,
                            ));
                            qualifiers
                                .push((TypeQualifier::Layout(StructLayout::Std430), token.meta));
                        }
                        "std140" => qualifiers
                            .push((TypeQualifier::Layout(StructLayout::Std140), token.meta)),
                        "std430" => qualifiers
                            .push((TypeQualifier::Layout(StructLayout::Std430), token.meta)),
                        "early_fragment_tests" => {
                            qualifiers.push((TypeQualifier::EarlyFragmentTests, token.meta))
                        }
                        _ => return Err(ErrorKind::UnknownLayoutQualifier(token.meta, name)),
                    }
                };

                Ok(())
            }
            // TODO: handle Shared?
            _ => Err(ErrorKind::InvalidToken(
                token,
                vec![ExpectedToken::Identifier],
            )),
        }
    }

    fn parse_constant_expression(&mut self) -> Result<(Handle<Constant>, SourceMetadata)> {
        let mut expressions = Arena::new();
        let mut locals = Arena::new();
        let mut arguments = Vec::new();
        let mut block = Block::new();

        let mut ctx = Context::new(
            self.program,
            &mut block,
            &mut expressions,
            &mut locals,
            &mut arguments,
        );

        let expr = self.parse_conditional(&mut ctx, &mut block, None)?;
        let (root, meta) = ctx.lower_expect(self.program, expr, false, &mut block)?;

        Ok((self.program.solve_constant(&ctx, root, meta)?, meta))
    }

    fn parse_external_declaration(&mut self) -> Result<()> {
        // TODO: Create body and expressions arena to be used in all entry
        // points to handle this case
        // ```glsl
        // // This is valid and the body of main will contain the assignment
        // float b;
        // float a = b = 1;
        //
        // void main() {}
        // ```
        let (mut e, mut l, mut a) = (Arena::new(), Arena::new(), Vec::new());
        let mut body = Block::new();
        let mut ctx = Context::new(self.program, &mut body, &mut e, &mut l, &mut a);

        if !self.parse_declaration(&mut ctx, &mut body, true)? {
            let token = self.bump()?;
            match token.value {
                TokenValue::Semicolon if self.program.version == 460 => Ok(()),
                _ => {
                    let expected = match self.program.version {
                        460 => vec![TokenValue::Semicolon.into(), ExpectedToken::Eof],
                        _ => vec![ExpectedToken::Eof],
                    };
                    Err(ErrorKind::InvalidToken(token, expected))
                }
            }
        } else {
            Ok(())
        }
    }

    fn peek_type_name(&mut self) -> bool {
        let program = &self.program;
        self.lexer.peek().map_or(false, |t| match t.value {
            TokenValue::TypeName(_) | TokenValue::Void => true,
            TokenValue::Struct => true,
            TokenValue::Identifier(ref ident) => program.lookup_type.contains_key(ident),
            _ => false,
        })
    }

    fn peek_parameter_qualifier(&mut self) -> bool {
        self.lexer.peek().map_or(false, |t| match t.value {
            TokenValue::In | TokenValue::Out | TokenValue::InOut | TokenValue::Const => true,
            _ => false,
        })
    }

    /// Returns the parsed `ParameterQualifier` or `ParameterQualifier::In`
    fn parse_parameter_qualifier(&mut self) -> ParameterQualifier {
        if self.peek_parameter_qualifier() {
            match self.bump().unwrap().value {
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

    fn parse_initializer(
        &mut self,
        ty: Handle<Type>,
        ctx: &mut Context,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, SourceMetadata)> {
        // initializer:
        //     assignment_expression
        //     LEFT_BRACE initializer_list RIGHT_BRACE
        //     LEFT_BRACE initializer_list COMMA RIGHT_BRACE
        //
        // initializer_list:
        //     initializer
        //     initializer_list COMMA initializer
        if let Some(Token { mut meta, .. }) = self.bump_if(TokenValue::LeftBrace) {
            // initializer_list
            let mut components = Vec::new();
            loop {
                // TODO: Change type
                components.push(self.parse_initializer(ty, ctx, body)?.0);

                let token = self.bump()?;
                match token.value {
                    TokenValue::Comma => {
                        if let Some(Token { meta: end_meta, .. }) =
                            self.bump_if(TokenValue::RightBrace)
                        {
                            meta = meta.union(&end_meta);
                            break;
                        }
                    }
                    TokenValue::RightBrace => {
                        meta = meta.union(&token.meta);
                        break;
                    }
                    _ => {
                        return Err(ErrorKind::InvalidToken(
                            token,
                            vec![TokenValue::Comma.into(), TokenValue::RightBrace.into()],
                        ))
                    }
                }
            }

            Ok((
                ctx.add_expression(Expression::Compose { ty, components }, body),
                meta,
            ))
        } else {
            let expr = self.parse_assignment(ctx, body)?;
            Ok(ctx.lower_expect(self.program, expr, false, body)?)
        }
    }

    // Note: caller preparsed the type and qualifiers
    // Note: caller skips this if the fallthrough token is not expected to be consumed here so this
    // produced Error::InvalidToken if it isn't consumed
    fn parse_init_declarator_list(
        &mut self,
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
            .or_else(|| self.lexer.peek())
            .filter(|t| t.value == TokenValue::Comma)
            .is_some()
        {
            fallthrough.take().or_else(|| self.lexer.next());
        }

        loop {
            let token = fallthrough
                .take()
                .ok_or(ErrorKind::EndOfFile)
                .or_else(|_| self.bump())?;
            let name = match token.value {
                TokenValue::Semicolon => break,
                TokenValue::Identifier(name) => name,
                _ => {
                    return Err(ErrorKind::InvalidToken(
                        token,
                        vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                    ))
                }
            };
            let mut meta = token.meta;

            // array_specifier
            // array_specifier EQUAL initializer
            // EQUAL initializer

            // parse an array specifier if it exists
            // NOTE: unlike other parse methods this one doesn't expect an array specifier and
            // returns Ok(None) rather than an error if there is not one
            let array_specifier = self.parse_array_specifier()?;
            let ty = self.maybe_array(ty, array_specifier);

            let init = self
                .bump_if(TokenValue::Assign)
                .map::<Result<_>, _>(|_| {
                    let (mut expr, init_meta) = self.parse_initializer(ty, ctx.ctx, ctx.body)?;

                    if let Some(kind) = self.program.module.types[ty].inner.scalar_kind() {
                        ctx.ctx
                            .implicit_conversion(self.program, &mut expr, init_meta, kind)?;
                    }

                    meta = meta.union(&init_meta);

                    Ok((expr, init_meta))
                })
                .transpose()?;

            // TODO: Should we try to make constants here?
            // This is mostly a hack because we don't yet support adding
            // bodies to entry points for variable initialization
            let maybe_constant =
                init.and_then(|(root, meta)| self.program.solve_constant(ctx.ctx, root, meta).ok());

            let pointer = ctx.add_var(self.program, ty, name, maybe_constant, meta)?;

            if let Some((value, _)) = init.filter(|_| maybe_constant.is_none()) {
                ctx.flush_expressions();
                ctx.body.push(Statement::Store { pointer, value });
            }

            let token = self.bump()?;
            match token.value {
                TokenValue::Semicolon => break,
                TokenValue::Comma => {}
                _ => {
                    return Err(ErrorKind::InvalidToken(
                        token,
                        vec![TokenValue::Comma.into(), TokenValue::Semicolon.into()],
                    ))
                }
            }
        }

        Ok(())
    }

    /// `external` whether or not we are in a global or local context
    fn parse_declaration(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        external: bool,
    ) -> Result<bool> {
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

        if self.peek_type_qualifier() || self.peek_type_name() {
            let qualifiers = self.parse_type_qualifiers()?;

            if self.peek_type_name() {
                // This branch handles variables and function prototypes and if
                // external is true also function definitions
                let (ty, mut meta) = self.parse_type()?;

                let token = self.bump()?;
                let token_fallthrough = match token.value {
                    TokenValue::Identifier(name) => match self.expect_peek()?.value {
                        TokenValue::LeftParen => {
                            // This branch handles function definition and prototypes
                            self.bump()?;

                            let result = ty.map(|ty| FunctionResult { ty, binding: None });
                            let mut expressions = Arena::new();
                            let mut local_variables = Arena::new();
                            let mut arguments = Vec::new();
                            let mut parameters = Vec::new();
                            let mut body = Block::new();
                            let mut sig = FunctionSignature {
                                name: name.clone(),
                                parameters: Vec::new(),
                            };

                            let mut context = Context::new(
                                self.program,
                                &mut body,
                                &mut expressions,
                                &mut local_variables,
                                &mut arguments,
                            );

                            self.parse_function_args(
                                &mut context,
                                &mut body,
                                &mut parameters,
                                &mut sig,
                            )?;

                            let end_meta = self.expect(TokenValue::RightParen)?.meta;
                            meta = meta.union(&end_meta);

                            let token = self.bump()?;
                            return match token.value {
                                TokenValue::Semicolon => {
                                    // This branch handles function prototypes
                                    self.program.add_prototype(
                                        Function {
                                            name: Some(name),
                                            result,
                                            arguments,
                                            ..Default::default()
                                        },
                                        sig,
                                        parameters,
                                        meta,
                                    )?;

                                    Ok(true)
                                }
                                TokenValue::LeftBrace if external => {
                                    // This branch handles function definitions
                                    // as you can see by the guard this branch
                                    // only happens if external is also true

                                    // parse the body
                                    self.parse_compound_statement(&mut context, &mut body)?;

                                    let Context { arg_use, .. } = context;
                                    let handle = self.program.add_function(
                                        Function {
                                            name: Some(name),
                                            result,
                                            expressions,
                                            named_expressions: crate::FastHashMap::default(),
                                            local_variables,
                                            arguments,
                                            body,
                                        },
                                        sig,
                                        parameters,
                                        meta,
                                    )?;

                                    self.program.function_arg_use[handle.index()] = arg_use;

                                    Ok(true)
                                }
                                _ if external => Err(ErrorKind::InvalidToken(
                                    token,
                                    vec![
                                        TokenValue::LeftBrace.into(),
                                        TokenValue::Semicolon.into(),
                                    ],
                                )),
                                _ => Err(ErrorKind::InvalidToken(
                                    token,
                                    vec![TokenValue::Semicolon.into()],
                                )),
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

                    self.parse_init_declarator_list(ty, Some(token_fallthrough), &mut ctx)?;
                } else {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Declaration cannot have void type".into(),
                    ));
                }

                Ok(true)
            } else {
                // This branch handles struct definitions and modifiers like
                // ```glsl
                // layout(early_fragment_tests);
                // ```
                let token = self.bump()?;
                match token.value {
                    TokenValue::Identifier(ty_name) => {
                        if self.bump_if(TokenValue::LeftBrace).is_some() {
                            self.parse_block_declaration(&qualifiers, ty_name, token.meta)
                        } else {
                            //TODO: declaration
                            // type_qualifier IDENTIFIER SEMICOLON
                            // type_qualifier IDENTIFIER identifier_list SEMICOLON
                            todo!()
                        }
                    }
                    TokenValue::Semicolon => {
                        for &(ref qualifier, meta) in qualifiers.iter() {
                            match *qualifier {
                                TypeQualifier::WorkGroupSize(i, value) => {
                                    self.program.workgroup_size[i] = value
                                }
                                TypeQualifier::EarlyFragmentTests => {
                                    self.program.early_fragment_tests = true;
                                }
                                TypeQualifier::StorageQualifier(_) => {
                                    // TODO: Maybe add some checks here
                                    // This is needed because of cases like
                                    // layout(early_fragment_tests) in;
                                }
                                _ => {
                                    return Err(ErrorKind::SemanticError(
                                        meta,
                                        "Qualifier not supported as standalone".into(),
                                    ));
                                }
                            }
                        }

                        Ok(true)
                    }
                    _ => Err(ErrorKind::InvalidToken(
                        token,
                        vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                    )),
                }
            }
        } else {
            Ok(false)
        }
    }

    fn parse_block_declaration(
        &mut self,
        qualifiers: &[(TypeQualifier, SourceMetadata)],
        ty_name: String,
        mut meta: SourceMetadata,
    ) -> Result<bool> {
        let mut members = Vec::new();
        let span = self.parse_struct_declaration_list(&mut members)?;
        self.expect(TokenValue::RightBrace)?;

        let mut ty = self.program.module.types.append(Type {
            name: Some(ty_name),
            inner: TypeInner::Struct {
                top_level: true,
                members: members.clone(),
                span,
            },
        });

        let token = self.bump()?;
        let name = match token.value {
            TokenValue::Semicolon => None,
            TokenValue::Identifier(name) => {
                if let Some(size) = self.parse_array_specifier()? {
                    ty = self.program.module.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Array {
                            base: ty,
                            size,
                            stride: self.program.module.types[ty]
                                .inner
                                .span(&self.program.module.constants),
                        },
                    });
                }

                self.expect(TokenValue::Semicolon)?;

                Some(name)
            }
            _ => {
                return Err(ErrorKind::InvalidToken(
                    token,
                    vec![ExpectedToken::Identifier, TokenValue::Semicolon.into()],
                ))
            }
        };
        meta = meta.union(&token.meta);

        let global = self.program.add_global_var(VarDeclaration {
            qualifiers,
            ty,
            name,
            init: None,
            meta,
        })?;

        for (i, k) in members
            .into_iter()
            .enumerate()
            .filter_map(|(i, m)| m.name.map(|s| (i as u32, s)))
        {
            self.program.global_variables.push((
                k,
                GlobalLookup {
                    kind: match global {
                        GlobalOrConstant::Global(handle) => {
                            GlobalLookupKind::BlockSelect(handle, i)
                        }
                        GlobalOrConstant::Constant(handle) => GlobalLookupKind::Constant(handle),
                    },
                    entry_arg: None,
                    mutable: true,
                },
            ));
        }

        Ok(true)
    }

    // TODO: Accept layout arguments
    fn parse_struct_declaration_list(&mut self, members: &mut Vec<StructMember>) -> Result<u32> {
        let mut span = 0;

        loop {
            // TODO: type_qualifier

            let ty = self.parse_type_non_void()?.0;
            let name = self.expect_ident()?.0;

            let array_specifier = self.parse_array_specifier()?;
            let ty = self.maybe_array(ty, array_specifier);

            self.expect(TokenValue::Semicolon)?;

            members.push(StructMember {
                name: Some(name),
                ty,
                binding: None,
                offset: span,
            });

            span += self.program.module.types[ty]
                .inner
                .span(&self.program.module.constants);

            if let TokenValue::RightBrace = self.expect_peek()?.value {
                break;
            }
        }

        Ok(span)
    }

    fn parse_primary(&mut self, ctx: &mut Context, body: &mut Block) -> Result<Handle<HirExpr>> {
        let mut token = self.bump()?;

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
                let expr = self.parse_expression(ctx, body)?;
                let meta = self.expect(TokenValue::RightParen)?.meta;

                token.meta = token.meta.union(&meta);

                return Ok(expr);
            }
            _ => {
                return Err(ErrorKind::InvalidToken(
                    token,
                    vec![
                        TokenValue::LeftParen.into(),
                        ExpectedToken::IntLiteral,
                        ExpectedToken::FloatLiteral,
                        ExpectedToken::BoolLiteral,
                    ],
                ))
            }
        };

        let handle = self.program.module.constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar { width, value },
        });

        Ok(ctx.hir_exprs.append(HirExpr {
            kind: HirExprKind::Constant(handle),
            meta: token.meta,
        }))
    }

    fn parse_function_call_args(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        meta: &mut SourceMetadata,
    ) -> Result<Vec<Handle<HirExpr>>> {
        let mut args = Vec::new();
        if let Some(token) = self.bump_if(TokenValue::RightParen) {
            *meta = meta.union(&token.meta);
        } else {
            loop {
                args.push(self.parse_assignment(ctx, body)?);

                let token = self.bump()?;
                match token.value {
                    TokenValue::Comma => {}
                    TokenValue::RightParen => {
                        *meta = meta.union(&token.meta);
                        break;
                    }
                    _ => {
                        return Err(ErrorKind::InvalidToken(
                            token,
                            vec![TokenValue::Comma.into(), TokenValue::RightParen.into()],
                        ))
                    }
                }
            }
        }

        Ok(args)
    }

    fn parse_postfix(&mut self, ctx: &mut Context, body: &mut Block) -> Result<Handle<HirExpr>> {
        let mut base = match self.expect_peek()?.value {
            TokenValue::Identifier(_) => {
                let (name, mut meta) = self.expect_ident()?;

                let expr = if self.bump_if(TokenValue::LeftParen).is_some() {
                    let args = self.parse_function_call_args(ctx, body, &mut meta)?;

                    let kind = match self.program.lookup_type.get(&name) {
                        Some(ty) => FunctionCallKind::TypeConstructor(*ty),
                        None => FunctionCallKind::Function(name),
                    };

                    HirExpr {
                        kind: HirExprKind::Call(FunctionCall { kind, args }),
                        meta,
                    }
                } else {
                    let var = match self.program.lookup_variable(ctx, body, &name)? {
                        Some(var) => var,
                        None => return Err(ErrorKind::UnknownVariable(meta, name)),
                    };

                    HirExpr {
                        kind: HirExprKind::Variable(var),
                        meta,
                    }
                };

                ctx.hir_exprs.append(expr)
            }
            TokenValue::TypeName(_) => {
                let Token { value, mut meta } = self.bump()?;

                let handle = if let TokenValue::TypeName(ty) = value {
                    self.program.module.types.fetch_or_append(ty)
                } else {
                    unreachable!()
                };

                self.expect(TokenValue::LeftParen)?;
                let args = self.parse_function_call_args(ctx, body, &mut meta)?;

                ctx.hir_exprs.append(HirExpr {
                    kind: HirExprKind::Call(FunctionCall {
                        kind: FunctionCallKind::TypeConstructor(handle),
                        args,
                    }),
                    meta,
                })
            }
            _ => self.parse_primary(ctx, body)?,
        };

        while let TokenValue::LeftBracket
        | TokenValue::Dot
        | TokenValue::Increment
        | TokenValue::Decrement = self.expect_peek()?.value
        {
            let Token { value, meta } = self.bump()?;

            match value {
                TokenValue::LeftBracket => {
                    let index = self.parse_expression(ctx, body)?;
                    let end_meta = self.expect(TokenValue::RightBracket)?.meta;

                    base = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::Access { base, index },
                        meta: meta.union(&end_meta),
                    })
                }
                TokenValue::Dot => {
                    let (field, end_meta) = self.expect_ident()?;

                    base = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::Select { base, field },
                        meta: meta.union(&end_meta),
                    })
                }
                TokenValue::Increment => {
                    base = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::IncDec {
                            increment: true,
                            postfix: true,
                            expr: base,
                        },
                        meta,
                    })
                }
                TokenValue::Decrement => {
                    base = ctx.hir_exprs.append(HirExpr {
                        kind: HirExprKind::IncDec {
                            increment: false,
                            postfix: true,
                            expr: base,
                        },
                        meta,
                    })
                }
                _ => unreachable!(),
            }
        }

        Ok(base)
    }

    fn parse_unary(&mut self, ctx: &mut Context, body: &mut Block) -> Result<Handle<HirExpr>> {
        // TODO: prefix inc/dec
        Ok(match self.expect_peek()?.value {
            TokenValue::Plus | TokenValue::Dash | TokenValue::Bang | TokenValue::Tilde => {
                let Token { value, meta } = self.bump()?;

                let expr = self.parse_unary(ctx, body)?;
                let end_meta = ctx.hir_exprs[expr].meta;

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

                ctx.hir_exprs.append(HirExpr {
                    kind,
                    meta: meta.union(&end_meta),
                })
            }
            TokenValue::Increment | TokenValue::Decrement => {
                let Token { value, meta } = self.bump()?;

                let expr = self.parse_unary(ctx, body)?;

                ctx.hir_exprs.append(HirExpr {
                    kind: HirExprKind::IncDec {
                        increment: match value {
                            TokenValue::Increment => true,
                            TokenValue::Decrement => false,
                            _ => unreachable!(),
                        },
                        postfix: false,
                        expr,
                    },
                    meta,
                })
            }
            _ => self.parse_postfix(ctx, body)?,
        })
    }

    fn parse_binary(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        passtrough: Option<Handle<HirExpr>>,
        min_bp: u8,
    ) -> Result<Handle<HirExpr>> {
        let mut left = passtrough
            .ok_or(ErrorKind::EndOfFile /* Dummy error */)
            .or_else(|_| self.parse_unary(ctx, body))?;
        let start_meta = ctx.hir_exprs[left].meta;

        while let Some((l_bp, r_bp)) = binding_power(&self.expect_peek()?.value) {
            if l_bp < min_bp {
                break;
            }

            let Token { value, .. } = self.bump()?;

            let right = self.parse_binary(ctx, body, None, r_bp)?;
            let end_meta = ctx.hir_exprs[right].meta;

            left = ctx.hir_exprs.append(HirExpr {
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
                meta: start_meta.union(&end_meta),
            })
        }

        Ok(left)
    }

    fn parse_conditional(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        passtrough: Option<Handle<HirExpr>>,
    ) -> Result<Handle<HirExpr>> {
        let mut condition = self.parse_binary(ctx, body, passtrough, 0)?;
        let start_meta = ctx.hir_exprs[condition].meta;

        if self.bump_if(TokenValue::Question).is_some() {
            let accept = self.parse_expression(ctx, body)?;
            self.expect(TokenValue::Colon)?;
            let reject = self.parse_assignment(ctx, body)?;
            let end_meta = ctx.hir_exprs[reject].meta;

            condition = ctx.hir_exprs.append(HirExpr {
                kind: HirExprKind::Conditional {
                    condition,
                    accept,
                    reject,
                },
                meta: start_meta.union(&end_meta),
            })
        }

        Ok(condition)
    }

    fn parse_assignment(&mut self, ctx: &mut Context, body: &mut Block) -> Result<Handle<HirExpr>> {
        let tgt = self.parse_unary(ctx, body)?;
        let start_meta = ctx.hir_exprs[tgt].meta;

        Ok(match self.expect_peek()?.value {
            TokenValue::Assign => {
                self.bump()?;
                let value = self.parse_assignment(ctx, body)?;
                let end_meta = ctx.hir_exprs[value].meta;

                ctx.hir_exprs.append(HirExpr {
                    kind: HirExprKind::Assign { tgt, value },
                    meta: start_meta.union(&end_meta),
                })
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
                let token = self.bump()?;
                let right = self.parse_assignment(ctx, body)?;
                let end_meta = ctx.hir_exprs[right].meta;

                let value = ctx.hir_exprs.append(HirExpr {
                    meta: start_meta.union(&end_meta),
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
                });

                ctx.hir_exprs.append(HirExpr {
                    kind: HirExprKind::Assign { tgt, value },
                    meta: start_meta.union(&end_meta),
                })
            }
            _ => self.parse_conditional(ctx, body, Some(tgt))?,
        })
    }

    fn parse_expression(&mut self, ctx: &mut Context, body: &mut Block) -> Result<Handle<HirExpr>> {
        let mut expr = self.parse_assignment(ctx, body)?;

        while let TokenValue::Comma = self.expect_peek()?.value {
            self.bump()?;
            expr = self.parse_assignment(ctx, body)?;
        }

        Ok(expr)
    }

    fn parse_statement(&mut self, ctx: &mut Context, body: &mut Block) -> Result<()> {
        // TODO: This prevents snippets like the following from working
        // ```glsl
        // vec4(1.0);
        // ```
        // But this would require us to add lookahead to also support
        // declarations and since this statement is very unlikely and most
        // likely an error, for now we don't support it
        if self.peek_type_name() || self.peek_type_qualifier() {
            self.parse_declaration(ctx, body, false)?;
            return Ok(());
        }

        match self.expect_peek()?.value {
            TokenValue::Continue => {
                self.bump()?;
                body.push(Statement::Continue);
                self.expect(TokenValue::Semicolon)?;
            }
            TokenValue::Break => {
                self.bump()?;
                body.push(Statement::Break);
                self.expect(TokenValue::Semicolon)?;
            }
            TokenValue::Return => {
                self.bump()?;
                let value = match self.expect_peek()?.value {
                    TokenValue::Semicolon => {
                        self.bump()?;
                        None
                    }
                    _ => {
                        // TODO: Implicit conversions
                        let expr = self.parse_expression(ctx, body)?;
                        self.expect(TokenValue::Semicolon)?;
                        Some(ctx.lower_expect(self.program, expr, false, body)?.0)
                    }
                };

                ctx.emit_flush(body);
                ctx.emit_start();

                body.push(Statement::Return { value })
            }
            TokenValue::Discard => {
                self.bump()?;
                body.push(Statement::Kill);
                self.expect(TokenValue::Semicolon)?;
            }
            TokenValue::If => {
                self.bump()?;

                self.expect(TokenValue::LeftParen)?;
                let condition = {
                    let expr = self.parse_expression(ctx, body)?;
                    ctx.lower_expect(self.program, expr, false, body)?.0
                };
                self.expect(TokenValue::RightParen)?;

                ctx.emit_flush(body);
                ctx.emit_start();

                let mut accept = Block::new();
                self.parse_statement(ctx, &mut accept)?;

                let mut reject = Block::new();
                if self.bump_if(TokenValue::Else).is_some() {
                    self.parse_statement(ctx, &mut reject)?;
                }

                body.push(Statement::If {
                    condition,
                    accept,
                    reject,
                });
            }
            TokenValue::Switch => {
                self.bump()?;

                self.expect(TokenValue::LeftParen)?;
                // TODO: Implicit conversions
                let selector = {
                    let expr = self.parse_expression(ctx, body)?;
                    ctx.lower_expect(self.program, expr, false, body)?.0
                };
                self.expect(TokenValue::RightParen)?;

                ctx.emit_flush(body);
                ctx.emit_start();

                let mut cases = Vec::new();
                let mut default = Block::new();

                self.expect(TokenValue::LeftBrace)?;
                loop {
                    match self.expect_peek()?.value {
                        TokenValue::Case => {
                            self.bump()?;
                            let value = {
                                let expr = self.parse_expression(ctx, body)?;
                                let (root, meta) =
                                    ctx.lower_expect(self.program, expr, false, body)?;
                                let constant = self.program.solve_constant(ctx, root, meta)?;

                                match self.program.module.constants[constant].inner {
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

                            self.expect(TokenValue::Colon)?;

                            let mut body = Block::new();

                            loop {
                                match self.expect_peek()?.value {
                                    TokenValue::Case
                                    | TokenValue::Default
                                    | TokenValue::RightBrace => break,
                                    _ => self.parse_statement(ctx, &mut body)?,
                                }
                            }

                            let fall_through = body.iter().any(|s| {
                                mem::discriminant(s) == mem::discriminant(&Statement::Break)
                            });

                            cases.push(SwitchCase {
                                value,
                                body,
                                fall_through,
                            })
                        }
                        TokenValue::Default => {
                            let Token { meta, .. } = self.bump()?;
                            self.expect(TokenValue::Colon)?;

                            if !default.is_empty() {
                                return Err(ErrorKind::SemanticError(
                                    meta,
                                    "Can only have one default case per switch statement".into(),
                                ));
                            }

                            loop {
                                match self.expect_peek()?.value {
                                    TokenValue::Case | TokenValue::RightBrace => break,
                                    _ => self.parse_statement(ctx, &mut &mut default)?,
                                }
                            }
                        }
                        TokenValue::RightBrace => {
                            self.bump()?;
                            break;
                        }
                        _ => {
                            return Err(ErrorKind::InvalidToken(
                                self.bump()?,
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
                self.bump()?;

                let mut loop_body = Block::new();

                self.expect(TokenValue::LeftParen)?;
                let root = self.parse_expression(ctx, &mut loop_body)?;
                self.expect(TokenValue::RightParen)?;

                let expr = ctx
                    .lower_expect(self.program, root, false, &mut loop_body)?
                    .0;
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

                self.parse_statement(ctx, &mut loop_body)?;

                body.push(Statement::Loop {
                    body: loop_body,
                    continuing: Block::new(),
                })
            }
            TokenValue::Do => {
                self.bump()?;

                let mut loop_body = Block::new();
                self.parse_statement(ctx, &mut loop_body)?;

                self.expect(TokenValue::While)?;
                self.expect(TokenValue::LeftParen)?;
                let root = self.parse_expression(ctx, &mut loop_body)?;
                self.expect(TokenValue::RightParen)?;

                let expr = ctx
                    .lower_expect(self.program, root, false, &mut loop_body)?
                    .0;
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
                self.bump()?;

                ctx.push_scope();
                self.expect(TokenValue::LeftParen)?;

                if self.bump_if(TokenValue::Semicolon).is_none() {
                    if self.peek_type_name() || self.peek_type_qualifier() {
                        self.parse_declaration(ctx, body, false)?;
                    } else {
                        self.parse_expression(ctx, body)?;
                        self.expect(TokenValue::Semicolon)?;
                    }
                }

                ctx.emit_flush(body);
                ctx.emit_start();

                let (mut block, mut continuing) = (Block::new(), Block::new());

                if self.bump_if(TokenValue::Semicolon).is_none() {
                    let expr = if self.peek_type_name() || self.peek_type_qualifier() {
                        let qualifiers = self.parse_type_qualifiers()?;
                        let (ty, meta) = self.parse_type_non_void()?;
                        let name = self.expect_ident()?.0;

                        self.expect(TokenValue::Assign)?;

                        let (value, end_meta) = self.parse_initializer(ty, ctx, &mut block)?;

                        let decl = VarDeclaration {
                            qualifiers: &qualifiers,
                            ty,
                            name: Some(name),
                            init: None,
                            meta: meta.union(&end_meta),
                        };

                        let pointer = self.program.add_local_var(ctx, &mut block, decl)?;

                        ctx.emit_flush(&mut block);
                        ctx.emit_start();

                        block.push(Statement::Store { pointer, value });

                        value
                    } else {
                        let root = self.parse_expression(ctx, &mut block)?;
                        ctx.lower_expect(self.program, root, false, &mut block)?.0
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

                    self.expect(TokenValue::Semicolon)?;
                }

                match self.expect_peek()?.value {
                    TokenValue::RightParen => {}
                    _ => {
                        let rest = self.parse_expression(ctx, &mut continuing)?;
                        ctx.lower(self.program, rest, false, &mut continuing)?;
                    }
                }

                self.expect(TokenValue::RightParen)?;

                self.parse_statement(ctx, &mut block)?;

                body.push(Statement::Loop {
                    body: block,
                    continuing,
                });

                ctx.remove_current_scope();
            }
            TokenValue::LeftBrace => {
                self.bump()?;

                let mut block = Block::new();
                ctx.push_scope();

                self.parse_compound_statement(ctx, &mut block)?;

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
                let expr = self.parse_expression(ctx, body)?;
                ctx.lower(self.program, expr, false, body)?;
                self.expect(TokenValue::Semicolon)?;
            }
            TokenValue::Semicolon => {
                self.bump()?;
            }
            _ => {}
        }

        Ok(())
    }

    fn parse_compound_statement(&mut self, ctx: &mut Context, body: &mut Block) -> Result<()> {
        loop {
            if self.bump_if(TokenValue::RightBrace).is_some() {
                break;
            }

            self.parse_statement(ctx, body)?;
        }

        Ok(())
    }

    fn parse_function_args(
        &mut self,
        context: &mut Context,
        body: &mut Block,
        parameters: &mut Vec<ParameterQualifier>,
        sig: &mut FunctionSignature,
    ) -> Result<()> {
        loop {
            if self.peek_type_name() || self.peek_parameter_qualifier() {
                let qualifier = self.parse_parameter_qualifier();
                parameters.push(qualifier);
                let ty = self.parse_type_non_void()?.0;

                match self.expect_peek()?.value {
                    TokenValue::Comma => {
                        self.bump()?;
                        context.add_function_arg(&mut self.program, sig, body, None, ty, qualifier);
                        continue;
                    }
                    TokenValue::Identifier(_) => {
                        let name = self.expect_ident()?.0;

                        let size = self.parse_array_specifier()?;
                        let ty = self.maybe_array(ty, size);

                        context.add_function_arg(
                            &mut self.program,
                            sig,
                            body,
                            Some(name),
                            ty,
                            qualifier,
                        );

                        if self.bump_if(TokenValue::Comma).is_some() {
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

struct DeclarationContext<'ctx, 'fun> {
    qualifiers: Vec<(TypeQualifier, SourceMetadata)>,
    external: bool,

    ctx: &'ctx mut Context<'fun>,
    body: &'ctx mut Block,
}

impl<'ctx, 'fun> DeclarationContext<'ctx, 'fun> {
    fn add_var(
        &mut self,
        program: &mut Program,
        ty: Handle<Type>,
        name: String,
        init: Option<Handle<Constant>>,
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>> {
        let decl = VarDeclaration {
            qualifiers: &self.qualifiers,
            ty,
            name: Some(name),
            init,
            meta,
        };

        match self.external {
            true => {
                let global = program.add_global_var(decl)?;
                let expr = match global {
                    GlobalOrConstant::Global(handle) => Expression::GlobalVariable(handle),
                    GlobalOrConstant::Constant(handle) => Expression::Constant(handle),
                };
                Ok(self.ctx.add_expression(expr, self.body))
            }
            false => program.add_local_var(self.ctx, self.body, decl),
        }
    }

    fn flush_expressions(&mut self) {
        self.ctx.emit_flush(self.body);
        self.ctx.emit_start()
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
