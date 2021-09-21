use super::{
    ast::{FunctionKind, Profile, TypeQualifier},
    context::{Context, ExprPos},
    error::ExpectedToken,
    error::{Error, ErrorKind},
    lex::{Lexer, LexerResultKind},
    token::{Directive, DirectiveKind},
    token::{Token, TokenValue},
    variables::{GlobalOrConstant, VarDeclaration},
    Parser, Result,
};
use crate::{arena::Handle, Block, Constant, ConstantInner, Expression, ScalarValue, Span, Type};
use core::convert::TryFrom;
use pp_rs::token::{PreprocessorError, Token as PPToken, TokenValue as PPTokenValue};
use std::iter::Peekable;

mod declarations;
mod expressions;
mod functions;
mod types;

pub struct ParsingContext<'source> {
    lexer: Peekable<Lexer<'source>>,
    last_meta: Span,
}

impl<'source> ParsingContext<'source> {
    pub fn new(lexer: Lexer<'source>) -> Self {
        ParsingContext {
            lexer: lexer.peekable(),
            last_meta: Span::default(),
        }
    }

    pub fn expect_ident(&mut self, parser: &mut Parser) -> Result<(String, Span)> {
        let token = self.bump(parser)?;

        match token.value {
            TokenValue::Identifier(name) => Ok((name, token.meta)),
            _ => Err(Error {
                kind: ErrorKind::InvalidToken(token.value, vec![ExpectedToken::Identifier]),
                meta: token.meta,
            }),
        }
    }

    pub fn expect(&mut self, parser: &mut Parser, value: TokenValue) -> Result<Token> {
        let token = self.bump(parser)?;

        if token.value != value {
            Err(Error {
                kind: ErrorKind::InvalidToken(token.value, vec![value.into()]),
                meta: token.meta,
            })
        } else {
            Ok(token)
        }
    }

    pub fn next(&mut self, parser: &mut Parser) -> Option<Token> {
        loop {
            let res = self.lexer.next()?;

            match res.kind {
                LexerResultKind::Token(token) => {
                    self.last_meta = token.meta;
                    break Some(token);
                }
                LexerResultKind::Directive(directive) => {
                    parser.handle_directive(directive, res.meta)
                }
                LexerResultKind::Error(error) => parser.errors.push(Error {
                    kind: ErrorKind::PreprocessorError(error),
                    meta: res.meta,
                }),
            }
        }
    }

    pub fn bump(&mut self, parser: &mut Parser) -> Result<Token> {
        self.next(parser).ok_or(Error {
            kind: ErrorKind::EndOfFile,
            meta: self.last_meta,
        })
    }

    /// Returns None on the end of the file rather than an error like other methods
    pub fn bump_if(&mut self, parser: &mut Parser, value: TokenValue) -> Option<Token> {
        if self.peek(parser).filter(|t| t.value == value).is_some() {
            self.bump(parser).ok()
        } else {
            None
        }
    }

    pub fn peek(&mut self, parser: &mut Parser) -> Option<&Token> {
        match self.lexer.peek()?.kind {
            LexerResultKind::Token(_) => {
                let res = self.lexer.peek()?;

                match res.kind {
                    LexerResultKind::Token(ref token) => Some(token),
                    _ => unreachable!(),
                }
            }
            LexerResultKind::Error(_) | LexerResultKind::Directive(_) => {
                let res = self.lexer.next()?;

                match res.kind {
                    LexerResultKind::Directive(directive) => {
                        parser.handle_directive(directive, res.meta)
                    }
                    LexerResultKind::Error(error) => parser.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(error),
                        meta: res.meta,
                    }),
                    _ => unreachable!(),
                }

                self.peek(parser)
            }
        }
    }

    pub fn expect_peek(&mut self, parser: &mut Parser) -> Result<&Token> {
        let meta = self.last_meta;
        self.peek(parser).ok_or(Error {
            kind: ErrorKind::EndOfFile,
            meta,
        })
    }

    pub fn parse(&mut self, parser: &mut Parser) -> Result<()> {
        // Body and expression arena for global initialization
        let mut body = Block::new();
        let mut ctx = Context::new(parser, &mut body);

        while self.peek(parser).is_some() {
            self.parse_external_declaration(parser, &mut ctx, &mut body)?;
        }

        match parser.lookup_function.get("main").and_then(|declaration| {
            declaration
                .overloads
                .iter()
                .find_map(|decl| match decl.kind {
                    FunctionKind::Call(handle) if decl.defined && decl.parameters.is_empty() => {
                        Some(handle)
                    }
                    _ => None,
                })
        }) {
            Some(handle) => parser.add_entry_point(handle, body, ctx.expressions),
            None => parser.errors.push(Error {
                kind: ErrorKind::SemanticError("Missing entry point".into()),
                meta: Span::default(),
            }),
        }

        Ok(())
    }

    fn parse_uint_constant(&mut self, parser: &mut Parser) -> Result<(u32, Span)> {
        let (value, meta) = self.parse_constant_expression(parser)?;

        let int = match parser.module.constants[value].inner {
            ConstantInner::Scalar {
                value: ScalarValue::Uint(int),
                ..
            } => u32::try_from(int).map_err(|_| Error {
                kind: ErrorKind::SemanticError("int constant overflows".into()),
                meta,
            })?,
            ConstantInner::Scalar {
                value: ScalarValue::Sint(int),
                ..
            } => u32::try_from(int).map_err(|_| Error {
                kind: ErrorKind::SemanticError("int constant overflows".into()),
                meta,
            })?,
            _ => {
                return Err(Error {
                    kind: ErrorKind::SemanticError("Expected a uint constant".into()),
                    meta,
                })
            }
        };

        Ok((int, meta))
    }

    fn parse_constant_expression(
        &mut self,
        parser: &mut Parser,
    ) -> Result<(Handle<Constant>, Span)> {
        let mut block = Block::new();

        let mut ctx = Context::new(parser, &mut block);

        let mut stmt_ctx = ctx.stmt_ctx();
        let expr = self.parse_conditional(parser, &mut ctx, &mut stmt_ctx, &mut block, None)?;
        let (root, meta) = ctx.lower_expect(stmt_ctx, parser, expr, ExprPos::Rhs, &mut block)?;

        Ok((parser.solve_constant(&ctx, root, meta)?, meta))
    }
}

impl Parser {
    fn handle_directive(&mut self, directive: Directive, meta: Span) {
        let mut tokens = directive.tokens.into_iter();

        match directive.kind {
            DirectiveKind::Version { is_first_directive } => {
                if !is_first_directive {
                    self.errors.push(Error {
                        kind: ErrorKind::SemanticError(
                            "#version must occur first in shader".into(),
                        ),
                        meta,
                    })
                }

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Integer(int),
                        location,
                    }) => match int.value {
                        440 | 450 | 460 => self.meta.version = int.value as u16,
                        _ => self.errors.push(Error {
                            kind: ErrorKind::InvalidVersion(int.value),
                            meta: location.into(),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    }),
                    None => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedNewLine),
                        meta,
                    }),
                };

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Ident(name),
                        location,
                    }) => match name.as_str() {
                        "core" => self.meta.profile = Profile::Core,
                        _ => self.errors.push(Error {
                            kind: ErrorKind::InvalidProfile(name),
                            meta: location.into(),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    }),
                    None => {}
                };

                if let Some(PPToken { value, location }) = tokens.next() {
                    self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    })
                }
            }
            DirectiveKind::Extension => {
                // TODO: Proper extension handling
                // - Checking for extension support in the compiler
                // - Handle behaviors such as warn
                // - Handle the all extension
                let name = match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Ident(name),
                        ..
                    }) => Some(name),
                    Some(PPToken { value, location }) => {
                        self.errors.push(Error {
                            kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                                value,
                            )),
                            meta: location.into(),
                        });

                        None
                    }
                    None => {
                        self.errors.push(Error {
                            kind: ErrorKind::PreprocessorError(
                                PreprocessorError::UnexpectedNewLine,
                            ),
                            meta,
                        });

                        None
                    }
                };

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Punct(pp_rs::token::Punct::Colon),
                        ..
                    }) => {}
                    Some(PPToken { value, location }) => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    }),
                    None => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedNewLine),
                        meta,
                    }),
                };

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Ident(behavior),
                        location,
                    }) => match behavior.as_str() {
                        "require" | "enable" | "warn" | "disable" => {
                            if let Some(name) = name {
                                self.meta.extensions.insert(name);
                            }
                        }
                        _ => self.errors.push(Error {
                            kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                                PPTokenValue::Ident(behavior),
                            )),
                            meta: location.into(),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    }),
                    None => self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedNewLine),
                        meta,
                    }),
                }

                if let Some(PPToken { value, location }) = tokens.next() {
                    self.errors.push(Error {
                        kind: ErrorKind::PreprocessorError(PreprocessorError::UnexpectedToken(
                            value,
                        )),
                        meta: location.into(),
                    })
                }
            }
            DirectiveKind::Pragma => {
                // TODO: handle some common pragmas?
            }
        }
    }
}

pub struct DeclarationContext<'ctx> {
    qualifiers: Vec<(TypeQualifier, Span)>,
    external: bool,

    ctx: &'ctx mut Context,
    body: &'ctx mut Block,
}

impl<'ctx> DeclarationContext<'ctx> {
    fn add_var(
        &mut self,
        parser: &mut Parser,
        ty: Handle<Type>,
        name: String,
        init: Option<Handle<Constant>>,
        meta: Span,
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
                let global = parser.add_global_var(self.ctx, self.body, decl)?;
                let expr = match global {
                    GlobalOrConstant::Global(handle) => Expression::GlobalVariable(handle),
                    GlobalOrConstant::Constant(handle) => Expression::Constant(handle),
                };
                Ok(self.ctx.add_expression(expr, meta, self.body))
            }
            false => parser.add_local_var(self.ctx, self.body, decl),
        }
    }

    fn flush_expressions(&mut self) {
        self.ctx.emit_flush(self.body);
        self.ctx.emit_start()
    }
}
