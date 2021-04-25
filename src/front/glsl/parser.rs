use super::{
    ast::Profile,
    error::ErrorKind,
    lex::Lexer,
    token::{Token, TokenMetadata, TokenValue},
    Program,
};
use std::iter::Peekable;

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

    pub fn parse(&mut self) -> Result<(), ErrorKind> {
        self.parse_version()?;

        if let Some(token) = self.lexer.next() {
            Err(ErrorKind::InvalidToken(token))
        } else {
            Ok(())
        }
    }

    fn parse_version(&mut self) -> Result<(), ErrorKind> {
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

    fn expect_ident(&mut self) -> Result<(String, TokenMetadata), ErrorKind> {
        let token = self.bump()?;

        match token.value {
            TokenValue::Identifier(name) => Ok((name, token.meta)),
            _ => Err(ErrorKind::InvalidToken(token)),
        }
    }

    fn expect(&mut self, value: TokenValue) -> Result<Token, ErrorKind> {
        let token = self.bump()?;

        if token.value != value {
            Err(ErrorKind::InvalidToken(token))
        } else {
            Ok(token)
        }
    }

    fn bump(&mut self) -> Result<Token, ErrorKind> {
        self.lexer.next().ok_or(ErrorKind::EndOfFile)
    }
}
