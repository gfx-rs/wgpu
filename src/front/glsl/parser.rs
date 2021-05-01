use super::{
    ast::{FunctionContext, Profile, StorageQualifier, TypeQualifier},
    error::ErrorKind,
    lex::Lexer,
    token::{Token, TokenMetadata, TokenValue},
    Program,
};
use crate::{arena::Handle, ArraySize, Function, FunctionResult, StorageClass, Type, TypeInner};
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

    fn expect_peek(&mut self) -> Result<&Token> {
        self.lexer.peek().ok_or(ErrorKind::EndOfFile)
    }

    pub fn parse(&mut self) -> Result<()> {
        self.parse_version()?;

        while self.lexer.peek().is_some() {
            self.parse_external_declaration()?;
        }

        Ok(())
    }

    fn parse_version(&mut self) -> Result<()> {
        self.expect(TokenValue::Version)?;

        let version = self.bump()?;
        match version.value {
            TokenValue::IntConstant(i) => match i {
                440 | 450 | 460 => self.program.version = i as u16,
                _ => return Err(ErrorKind::InvalidVersion(version.meta, i)),
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
            | TokenValue::InOut
            | TokenValue::Uniform => true,
            _ => false,
        })
    }

    fn parse_type_qualifiers(&mut self) -> Result<Vec<TypeQualifier>> {
        let mut qualifiers = Vec::new();

        if self.peek_type_qualifier() {
            let token = self.bump()?;
            qualifiers.push(match token.value {
                TokenValue::Interpolation(i) => TypeQualifier::Interpolation(i),
                TokenValue::Const => TypeQualifier::StorageQualifier(StorageQualifier::Const),
                TokenValue::In => TypeQualifier::StorageQualifier(StorageQualifier::Input),
                TokenValue::Out => TypeQualifier::StorageQualifier(StorageQualifier::Output),
                TokenValue::InOut => TypeQualifier::StorageQualifier(StorageQualifier::InOut),
                TokenValue::Uniform => TypeQualifier::StorageQualifier(
                    StorageQualifier::StorageClass(StorageClass::Uniform),
                ),
                _ => unreachable!(),
            })
        }

        Ok(qualifiers)
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
    }

    fn peek_type_name(&mut self) -> bool {
        self.lexer.peek().map_or(false, |t| match t.value {
            TokenValue::TypeName(_) | TokenValue::Void => true,
            _ => false,
        })
    }

    /// `external` wheter or not we are in a global or local context
    fn parse_declaration(&mut self, external: bool) -> Result<bool> {
        // TODO: Handle precision qualifiers
        if self.peek_type_qualifier() || self.peek_type_name() {
            // TODO: Use qualifiers
            let _qualifiers = self.parse_type_qualifiers()?;

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
                                // Function prototypes
                                TokenValue::Semicolon => todo!(),
                                TokenValue::LeftBrace if external => {
                                    // TODO: body
                                    self.expect(TokenValue::RightBrace)?;

                                    Ok(true)
                                }
                                _ => Err(ErrorKind::InvalidToken(token)),
                            }
                        }
                        // Variable Declaration
                        TokenValue::Semicolon => todo!(),
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

                        if self.expect_peek()?.value == TokenValue::Comma {
                            self.bump()?;
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
