use crate::{
    arena::Handle,
    front::glsl::{
        ast::{Profile, TypeQualifier},
        context::Context,
        error::ErrorKind,
        error::ExpectedToken,
        lex::{Lexer, LexerResultKind},
        token::{Directive, DirectiveKind},
        token::{SourceMetadata, Token, TokenValue},
        variables::{GlobalOrConstant, VarDeclaration},
        ParseError, Parser, Result,
    },
    Block, Constant, ConstantInner, Expression, ScalarValue, Type,
};
use core::convert::TryFrom;
use pp_rs::token::{PreprocessorError, Token as PPToken, TokenValue as PPTokenValue};
use std::iter::Peekable;

mod declarations;
mod expressions;
mod functions;
mod types;

pub struct ParsingContext<'source> {
    lexer: Peekable<Lexer<'source>>,
}

impl<'source> ParsingContext<'source> {
    pub fn new(lexer: Lexer<'source>) -> Self {
        ParsingContext {
            lexer: lexer.peekable(),
        }
    }

    pub fn expect_ident(&mut self, parser: &mut Parser) -> Result<(String, SourceMetadata)> {
        let token = self.bump(parser)?;

        match token.value {
            TokenValue::Identifier(name) => Ok((name, token.meta)),
            _ => Err(ErrorKind::InvalidToken(
                token,
                vec![ExpectedToken::Identifier],
            )),
        }
    }

    pub fn expect(&mut self, parser: &mut Parser, value: TokenValue) -> Result<Token> {
        let token = self.bump(parser)?;

        if token.value != value {
            Err(ErrorKind::InvalidToken(token, vec![value.into()]))
        } else {
            Ok(token)
        }
    }

    pub fn next(&mut self, parser: &mut Parser) -> Option<Token> {
        loop {
            let res = self.lexer.next()?;

            match res.kind {
                LexerResultKind::Token(token) => break Some(token),
                LexerResultKind::Directive(directive) => {
                    parser.handle_directive(directive, res.meta)
                }
                LexerResultKind::Error(error) => parser.errors.push(ParseError {
                    kind: ErrorKind::PreprocessorError(res.meta, error),
                }),
            }
        }
    }

    pub fn bump(&mut self, parser: &mut Parser) -> Result<Token> {
        self.next(parser).ok_or(ErrorKind::EndOfFile)
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
                    LexerResultKind::Error(error) => parser.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(res.meta, error),
                    }),
                    _ => unreachable!(),
                }

                self.peek(parser)
            }
        }
    }

    pub fn expect_peek(&mut self, parser: &mut Parser) -> Result<&Token> {
        self.peek(parser).ok_or(ErrorKind::EndOfFile)
    }

    pub fn parse(&mut self, parser: &mut Parser) -> Result<()> {
        // Body and expression arena for global initialization
        let mut body = Block::new();
        let mut ctx = Context::new(parser, &mut body);

        while self.peek(parser).is_some() {
            self.parse_external_declaration(parser, &mut ctx, &mut body)?;
        }

        let handle = parser
            .lookup_function
            .get("main")
            .and_then(|declarations| {
                declarations
                    .iter()
                    .find(|decl| decl.defined && decl.parameters.is_empty())
                    .map(|decl| decl.handle)
            })
            .ok_or_else(|| {
                ErrorKind::SemanticError(SourceMetadata::default(), "Missing entry point".into())
            })?;

        parser.add_entry_point(handle, body, ctx.expressions)?;

        Ok(())
    }

    fn parse_uint_constant(&mut self, parser: &mut Parser) -> Result<(u32, SourceMetadata)> {
        let (value, meta) = self.parse_constant_expression(parser)?;

        let int = match parser.module.constants[value].inner {
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

    fn parse_constant_expression(
        &mut self,
        parser: &mut Parser,
    ) -> Result<(Handle<Constant>, SourceMetadata)> {
        let mut block = Block::new();

        let mut ctx = Context::new(parser, &mut block);

        let expr = self.parse_conditional(parser, &mut ctx, &mut block, None)?;
        let (root, meta) = ctx.lower_expect(parser, expr, false, &mut block)?;

        Ok((parser.solve_constant(&ctx, root, meta)?, meta))
    }
}

impl Parser {
    fn handle_directive(&mut self, directive: Directive, meta: SourceMetadata) {
        let mut tokens = directive.tokens.into_iter();

        match directive.kind {
            DirectiveKind::Version { is_first_directive } => {
                if !is_first_directive {
                    self.errors.push(ParseError {
                        kind: ErrorKind::SemanticError(
                            meta,
                            "#version must occur first in shader".into(),
                        ),
                    })
                }

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Integer(int),
                        location,
                    }) => match int.value {
                        440 | 450 | 460 => self.meta.version = int.value as u16,
                        _ => self.errors.push(ParseError {
                            kind: ErrorKind::InvalidVersion(location.into(), int.value),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
                    }),
                    None => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            meta,
                            PreprocessorError::UnexpectedNewLine,
                        ),
                    }),
                };

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Ident(name),
                        location,
                    }) => match name.as_str() {
                        "core" => self.meta.profile = Profile::Core,
                        _ => self.errors.push(ParseError {
                            kind: ErrorKind::InvalidProfile(location.into(), name),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
                    }),
                    None => {}
                };

                if let Some(PPToken { value, location }) = tokens.next() {
                    self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
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
                        self.errors.push(ParseError {
                            kind: ErrorKind::PreprocessorError(
                                location.into(),
                                PreprocessorError::UnexpectedToken(value),
                            ),
                        });

                        None
                    }
                    None => {
                        self.errors.push(ParseError {
                            kind: ErrorKind::PreprocessorError(
                                meta,
                                PreprocessorError::UnexpectedNewLine,
                            ),
                        });

                        None
                    }
                };

                match tokens.next() {
                    Some(PPToken {
                        value: PPTokenValue::Punct(pp_rs::token::Punct::Colon),
                        ..
                    }) => {}
                    Some(PPToken { value, location }) => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
                    }),
                    None => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            meta,
                            PreprocessorError::UnexpectedNewLine,
                        ),
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
                        _ => self.errors.push(ParseError {
                            kind: ErrorKind::PreprocessorError(
                                location.into(),
                                PreprocessorError::UnexpectedToken(PPTokenValue::Ident(behavior)),
                            ),
                        }),
                    },
                    Some(PPToken { value, location }) => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
                    }),
                    None => self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            meta,
                            PreprocessorError::UnexpectedNewLine,
                        ),
                    }),
                }

                if let Some(PPToken { value, location }) = tokens.next() {
                    self.errors.push(ParseError {
                        kind: ErrorKind::PreprocessorError(
                            location.into(),
                            PreprocessorError::UnexpectedToken(value),
                        ),
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
    qualifiers: Vec<(TypeQualifier, SourceMetadata)>,
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
                let global = parser.add_global_var(self.ctx, self.body, decl)?;
                let expr = match global {
                    GlobalOrConstant::Global(handle) => Expression::GlobalVariable(handle),
                    GlobalOrConstant::Constant(handle) => Expression::Constant(handle),
                };
                Ok(self.ctx.add_expression(expr, self.body))
            }
            false => parser.add_local_var(self.ctx, self.body, decl),
        }
    }

    fn flush_expressions(&mut self) {
        self.ctx.emit_flush(self.body);
        self.ctx.emit_start()
    }
}
