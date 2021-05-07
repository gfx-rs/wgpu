use super::{
    ast::{
        Expr, ExprKind, FunctionContext, Profile, StorageQualifier, StructLayout, TypeQualifier,
    },
    error::ErrorKind,
    lex::Lexer,
    token::{Token, TokenMetadata, TokenValue},
    Program,
};
use crate::{
    arena::Handle, ArraySize, BinaryOperator, Constant, ConstantInner, Function, FunctionResult,
    ResourceBinding, ScalarValue, Statement, StorageClass, Type, TypeInner, UnaryOperator,
};
use core::convert::TryFrom;
use std::iter::Peekable;

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

    fn expect_ident(&mut self) -> Result<(String, TokenMetadata)> {
        let token = self.bump()?;

        match token.value {
            TokenValue::Identifier(name) => Ok((name, token.meta)),
            _ => Err(ErrorKind::InvalidToken(token)),
        }
    }

    fn expect(&mut self, value: TokenValue) -> Result<Token> {
        let token = self.bump()?;

        if token.value != value {
            Err(ErrorKind::InvalidToken(token))
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
            _ => return Err(ErrorKind::InvalidToken(version)),
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

    fn parse_array_specifier(&mut self) -> Result<Option<ArraySize>> {
        // TODO: expressions
        if let Some(&TokenValue::LeftBracket) = self.lexer.peek().map(|t| &t.value) {
            self.bump()?;

            self.expect(TokenValue::RightBracket)?;

            Ok(Some(ArraySize::Dynamic))
        } else {
            Ok(None)
        }
    }

    fn parse_type(&mut self) -> Result<Option<Handle<Type>>> {
        let token = self.bump()?;
        let ty = match token.value {
            TokenValue::Void => None,
            TokenValue::TypeName(ty) => Some(ty),
            TokenValue::Struct => todo!(),
            _ => return Err(ErrorKind::InvalidToken(token)),
        };
        let handle = ty.map(|t| self.program.module.types.append(t));

        let size = self.parse_array_specifier()?;
        Ok(handle.map(|ty| self.maybe_array(ty, size)))
    }

    fn parse_type_non_void(&mut self) -> Result<Handle<Type>> {
        self.parse_type()?
            .ok_or_else(|| ErrorKind::SemanticError("Type can't be void".into()))
    }

    fn maybe_array(&mut self, base: Handle<Type>, size: Option<ArraySize>) -> Handle<Type> {
        size.map(|size| {
            self.program.module.types.append(Type {
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
            | TokenValue::Const
            | TokenValue::In
            | TokenValue::Out
            | TokenValue::Uniform
            | TokenValue::Layout => true,
            _ => false,
        })
    }

    fn parse_type_qualifiers(&mut self) -> Result<Vec<TypeQualifier>> {
        let mut qualifiers = Vec::new();

        while self.peek_type_qualifier() {
            let token = self.bump()?;

            // Handle layout qualifiers outside the match since this can push multiple values
            if token.value == TokenValue::Layout {
                self.parse_layout_qualifier_id_list(&mut qualifiers)?;
                continue;
            }

            qualifiers.push(match token.value {
                TokenValue::Interpolation(i) => TypeQualifier::Interpolation(i),
                TokenValue::Const => TypeQualifier::StorageQualifier(StorageQualifier::Const),
                TokenValue::In => TypeQualifier::StorageQualifier(StorageQualifier::Input),
                TokenValue::Out => TypeQualifier::StorageQualifier(StorageQualifier::Output),
                TokenValue::Uniform => TypeQualifier::StorageQualifier(
                    StorageQualifier::StorageClass(StorageClass::Uniform),
                ),

                _ => unreachable!(),
            })
        }

        Ok(qualifiers)
    }

    fn parse_layout_qualifier_id_list(
        &mut self,
        qualifiers: &mut Vec<TypeQualifier>,
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
            (Some(group), Some(binding)) => {
                qualifiers.push(TypeQualifier::ResourceBinding(ResourceBinding {
                    group,
                    binding,
                }))
            }
            // Produce an error if we have one of group or binding but not the other
            // TODO: include at least line info in errors?
            (Some(_), None) => {
                return Err(ErrorKind::SemanticError(
                    "set specified with no binding".into(),
                ))
            }
            (None, Some(_)) => {
                return Err(ErrorKind::SemanticError(
                    "binding specified with no set".into(),
                ))
            }
            (None, None) => (),
        }

        Ok(())
    }

    fn parse_uint_constant(&mut self) -> Result<u32> {
        let value = self.parse_constant_expression()?;

        // TODO: better errors
        match self.program.module.constants[value].inner {
            ConstantInner::Scalar {
                value: ScalarValue::Uint(int),
                ..
            } => u32::try_from(int)
                .map_err(|_| ErrorKind::SemanticError("int constant overflows".into())),
            ConstantInner::Scalar {
                value: ScalarValue::Sint(int),
                ..
            } => u32::try_from(int)
                .map_err(|_| ErrorKind::SemanticError("int constant overflows".into())),
            _ => Err(ErrorKind::SemanticError("Expected a uint constant".into())),
        }
    }

    fn parse_layout_qualifier_id(
        &mut self,
        qualifiers: &mut Vec<TypeQualifier>,
        group: &mut Option<u32>,
        binding: &mut Option<u32>,
    ) -> Result<()> {
        // layout_qualifier_id:
        //     IDENTIFIER
        //     IDENTIFIER EQUAL constant_expression
        //     SHARED
        let token = self.bump()?;
        match token.value {
            TokenValue::Identifier(name) => {
                if self.bump_if(TokenValue::Assign).is_some() {
                    let value = self.parse_uint_constant()?;

                    match name.as_str() {
                        "location" => qualifiers.push(TypeQualifier::Location(value)),
                        "set" => *group = Some(value),
                        "binding" => *binding = Some(value),
                        _ => return Err(ErrorKind::UnknownLayoutQualifier(token.meta, name)),
                    }
                } else {
                    match name.as_str() {
                        "std140" => qualifiers.push(TypeQualifier::Layout(StructLayout::Std140)),
                        "early_fragment_tests" => {
                            qualifiers.push(TypeQualifier::EarlyFragmentTests)
                        }
                        _ => return Err(ErrorKind::UnknownLayoutQualifier(token.meta, name)),
                    }
                };

                Ok(())
            }
            // TODO: handle Shared?
            _ => Err(ErrorKind::InvalidToken(token)),
        }
    }

    fn parse_constant_expression(&mut self) -> Result<Handle<Constant>> {
        let mut function = Function::default();
        let mut ctx = FunctionContext::new(&mut function);

        let expr = self.parse_conditional(&mut ctx, None)?;
        let root = ctx.resolve(self.program, expr, false)?;

        self.program.solve_constant(&function.expressions, root)
    }

    fn parse_external_declaration(&mut self) -> Result<()> {
        if !self.parse_declaration(true)? {
            let token = self.bump()?;
            match token.value {
                TokenValue::Semicolon if self.program.version == 460 => Ok(()),
                _ => Err(ErrorKind::InvalidToken(token)),
            }
        } else {
            Ok(())
        }
        // TODO
    }

    fn peek_type_name(&mut self) -> bool {
        self.lexer.peek().map_or(false, |t| match t.value {
            TokenValue::TypeName(_) | TokenValue::Void => true,
            _ => false,
        })
    }

    /// `external` whether or not we are in a global or local context
    fn parse_declaration(&mut self, external: bool) -> Result<bool> {
        //declaration:
        //    function_prototype  SEMICOLON
        //    init_declarator_list SEMICOLON
        //    PRECISION precision_qualifier type_specifier SEMICOLON
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE SEMICOLON
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE IDENTIFIER SEMICOLON
        //    type_qualifier IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE IDENTIFIER array_specifier SEMICOLON
        //    type_qualifier SEMICOLON type_qualifier IDENTIFIER SEMICOLON
        //    type_qualifier IDENTIFIER identifier_list SEMICOLON
        // TODO: Handle precision qualifiers
        if self.peek_type_qualifier() || self.peek_type_name() {
            let qualifiers = self.parse_type_qualifiers()?;

            if self.peek_type_name() {
                // Functions and variable declarations
                let ty = self.parse_type()?;

                let token = self.bump()?;

                match token.value {
                    TokenValue::Semicolon => Ok(true),
                    TokenValue::Identifier(name) => match self.expect_peek()?.value {
                        // Function definition/prototype
                        TokenValue::LeftParen => {
                            self.bump()?;

                            let mut function = Function {
                                name: Some(name),
                                result: ty.map(|ty| FunctionResult { ty, binding: None }),
                                ..Default::default()
                            };
                            let mut context = FunctionContext::new(&mut function);

                            self.parse_function_args(&mut context)?;

                            self.expect(TokenValue::RightParen)?;

                            let token = self.bump()?;
                            match token.value {
                                // TODO: Function prototypes
                                TokenValue::Semicolon => todo!(),
                                TokenValue::LeftBrace if external => {
                                    self.parse_compound_statement(&mut context)?;
                                    self.program.add_function(function)?;

                                    Ok(true)
                                }
                                _ => Err(ErrorKind::InvalidToken(token)),
                            }
                        }
                        // Variable Declaration
                        TokenValue::Semicolon => {
                            self.bump()?;

                            if let Some(ty) = ty {
                                if external {
                                    self.program.add_global_var(qualifiers, ty, name, None)?;
                                } else {
                                    // TODO: local variables
                                }
                            } else {
                                return Err(ErrorKind::SemanticError(
                                    "Declaration cannot have void type".into(),
                                ));
                            }

                            Ok(true)
                        }
                        TokenValue::Comma => todo!(),
                        _ => Err(ErrorKind::InvalidToken(self.bump()?)),
                    },
                    _ => Err(ErrorKind::InvalidToken(token)),
                }
            } else {
                // Structs and modifiers
                let token = self.bump()?;
                match token.value {
                    TokenValue::Identifier(_) => todo!(),
                    TokenValue::Semicolon => Ok(true),
                    _ => Err(ErrorKind::InvalidToken(token)),
                }
            }
        } else {
            Ok(false)
        }
    }

    fn parse_primary(&mut self, ctx: &mut FunctionContext) -> Result<Expr> {
        let token = self.bump()?;

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
                let expr = self.parse_expression(ctx)?;
                self.expect(TokenValue::RightParen)?;

                return Ok(expr);
            }
            _ => return Err(ErrorKind::InvalidToken(token)),
        };

        let handle = self.program.module.constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar { width, value },
        });

        Ok(Expr {
            kind: ExprKind::Constant(handle),
            meta: token.meta,
        })
    }

    fn parse_postfix(&mut self, ctx: &mut FunctionContext) -> Result<Expr> {
        let mut expr = match self.expect_peek()?.value {
            TokenValue::Identifier(_) => {
                let (name, meta) = self.expect_ident()?;

                // TODO: function calls
                // let mut args = Vec::new();
                // while TokenValue::RightBracket != self.bump()?.value {
                //     args.push(self.parse_expression(ctx)?)
                // }

                let var = match self.program.lookup_variable(ctx, &name)? {
                    Some(var) => var,
                    None => return Err(ErrorKind::UnknownVariable(meta, name)),
                };

                Expr {
                    kind: ExprKind::Variable(var),
                    meta,
                }
            }
            _ => self.parse_primary(ctx)?,
        };

        // TODO: postfix inc/dec
        while let TokenValue::LeftBracket | TokenValue::Dot = self.expect_peek()?.value {
            let Token { value, meta } = self.bump()?;

            match value {
                TokenValue::LeftBracket => {
                    let index = Box::new(self.parse_expression(ctx)?);
                    self.expect(TokenValue::RightBracket)?;

                    // TODO: metadata unification
                    expr = Expr {
                        kind: ExprKind::Access {
                            base: Box::new(expr),
                            index,
                        },
                        meta,
                    }
                }
                TokenValue::Dot => {
                    let field = self.expect_ident()?.0;

                    // TODO: metadata unification
                    expr = Expr {
                        kind: ExprKind::Select {
                            base: Box::new(expr),
                            field,
                        },
                        meta,
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(expr)
    }

    fn parse_unary(&mut self, ctx: &mut FunctionContext) -> Result<Expr> {
        // TODO: prefix inc/dec
        match self.expect_peek()?.value {
            TokenValue::Plus | TokenValue::Dash | TokenValue::Bang | TokenValue::Tilde => {
                let Token { value, meta } = self.bump()?;

                let expr = self.parse_unary(ctx)?;

                let kind = match value {
                    TokenValue::Dash => ExprKind::Unary {
                        op: UnaryOperator::Negate,
                        expr: Box::new(expr),
                    },
                    TokenValue::Bang | TokenValue::Tilde => ExprKind::Unary {
                        op: UnaryOperator::Not,
                        expr: Box::new(expr),
                    },
                    _ => return Ok(expr),
                };

                // TODO: metadata unification
                Ok(Expr { kind, meta })
            }
            _ => self.parse_postfix(ctx),
        }
    }

    fn parse_binary(
        &mut self,
        ctx: &mut FunctionContext,
        passtrough: Option<Expr>,
        min_bp: u8,
    ) -> Result<Expr> {
        let mut expr = passtrough
            .ok_or(ErrorKind::EndOfFile /* Dummy error */)
            .or_else(|_| self.parse_unary(ctx))?;

        while let Some((l_bp, r_bp)) = binding_power(&self.expect_peek()?.value) {
            if l_bp < min_bp {
                break;
            }

            let Token { value, meta } = self.bump()?;

            let right = Box::new(self.parse_binary(ctx, None, r_bp)?);

            expr = Expr {
                kind: ExprKind::Binary {
                    left: Box::new(expr),
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
                        TokenValue::LeftAngle => BinaryOperator::Greater,
                        TokenValue::RightAngle => BinaryOperator::Less,
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
            }
        }

        Ok(expr)
    }

    fn parse_conditional(
        &mut self,
        ctx: &mut FunctionContext,
        passtrough: Option<Expr>,
    ) -> Result<Expr> {
        let mut condition = self.parse_binary(ctx, passtrough, 0)?;

        if let TokenValue::Question = self.expect_peek()?.value {
            let meta = self.bump()?.meta;

            let accept = Box::new(self.parse_expression(ctx)?);
            self.expect(TokenValue::Colon)?;
            let reject = Box::new(self.parse_assignment(ctx)?);

            condition = Expr {
                kind: ExprKind::Conditional {
                    condition: Box::new(condition),
                    accept,
                    reject,
                },
                meta,
            };
        }

        Ok(condition)
    }

    fn parse_assignment(&mut self, ctx: &mut FunctionContext) -> Result<Expr> {
        let pointer = self.parse_unary(ctx)?;

        if let TokenValue::Assign = self.expect_peek()?.value {
            let meta = self.bump()?.meta;

            let value = Box::new(self.parse_assignment(ctx)?);

            Ok(Expr {
                kind: ExprKind::Assign {
                    tgt: Box::new(pointer),
                    value,
                },
                meta,
            })
        } else {
            self.parse_conditional(ctx, Some(pointer))
        }
    }

    fn parse_expression(&mut self, ctx: &mut FunctionContext) -> Result<Expr> {
        let mut expr = self.parse_assignment(ctx)?;

        while let TokenValue::Comma = self.expect_peek()?.value {
            self.bump()?;
            expr = self.parse_assignment(ctx)?;
        }

        Ok(expr)
    }

    fn parse_compound_statement(&mut self, ctx: &mut FunctionContext) -> Result<()> {
        loop {
            let token = self.bump()?;

            match token.value {
                TokenValue::Continue => {
                    ctx.function.body.push(Statement::Continue);
                    self.expect(TokenValue::Semicolon)?;
                }
                TokenValue::Break => {
                    ctx.function.body.push(Statement::Break);
                    self.expect(TokenValue::Semicolon)?;
                }
                TokenValue::Return => {
                    let value = match self.expect_peek()?.value {
                        TokenValue::Semicolon => {
                            self.bump()?;
                            None
                        }
                        _ => {
                            let expr = self.parse_expression(ctx)?;
                            self.expect(TokenValue::Semicolon)?;
                            Some(ctx.resolve(self.program, expr, false)?)
                        }
                    };

                    ctx.function.body.push(Statement::Return { value })
                }
                TokenValue::Discard => {
                    ctx.function.body.push(Statement::Kill);
                    self.expect(TokenValue::Semicolon)?;
                }
                TokenValue::LeftBrace => {
                    ctx.push_scope();
                    self.parse_compound_statement(ctx)?;
                    ctx.remove_current_scope();
                }
                TokenValue::RightBrace => break,
                _ => {}
            }
        }

        Ok(())
    }

    fn parse_function_args(&mut self, context: &mut FunctionContext) -> Result<()> {
        loop {
            // TODO: parameter qualifier
            if self.peek_type_name() {
                let ty = self.parse_type_non_void()?;

                match self.expect_peek()?.value {
                    TokenValue::Comma => {
                        self.bump()?;
                        context.add_function_arg(None, ty);
                        continue;
                    }
                    TokenValue::Identifier(_) => {
                        let name = self.expect_ident()?.0;

                        let size = self.parse_array_specifier()?;
                        let ty = self.maybe_array(ty, size);

                        context.add_function_arg(Some(name), ty);

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
