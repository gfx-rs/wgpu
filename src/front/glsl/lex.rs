use super::{
    token::{SourceMetadata, Token, TokenValue},
    types::parse_type,
};
use crate::FastHashMap;
use pp_rs::{
    pp::Preprocessor,
    token::{Punct, Token as PPToken, TokenValue as PPTokenValue},
};
use std::collections::VecDeque;

pub struct Lexer<'a> {
    pp: Preprocessor<'a>,
    tokens: VecDeque<PPToken>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str, defines: &'a FastHashMap<String, String>) -> Self {
        let mut pp = Preprocessor::new(input);
        for (define, value) in defines {
            pp.add_define(define, value).unwrap(); //TODO: handle error
        }
        Lexer {
            pp,
            tokens: Default::default(),
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        let mut meta = SourceMetadata::default();
        let pp_token = match self.tokens.pop_front() {
            Some(t) => t,
            None => match self.pp.next()? {
                Ok(t) => t,
                Err((err, loc)) => {
                    meta.start = loc.start as usize;
                    meta.end = loc.end as usize;
                    return Some(Token {
                        value: TokenValue::Unknown(err),
                        meta,
                    });
                }
            },
        };

        meta.start = pp_token.location.start as usize;
        meta.end = pp_token.location.end as usize;
        let value = match pp_token.value {
            PPTokenValue::Extension(extension) => {
                for t in extension.tokens {
                    self.tokens.push_back(t);
                }
                TokenValue::Extension
            }
            PPTokenValue::Float(float) => TokenValue::FloatConstant(float),
            PPTokenValue::Ident(ident) => {
                match ident.as_str() {
                    "layout" => TokenValue::Layout,
                    "in" => TokenValue::In,
                    "out" => TokenValue::Out,
                    "uniform" => TokenValue::Uniform,
                    "buffer" => TokenValue::Buffer,
                    "flat" => TokenValue::Interpolation(crate::Interpolation::Flat),
                    "noperspective" => TokenValue::Interpolation(crate::Interpolation::Linear),
                    "smooth" => TokenValue::Interpolation(crate::Interpolation::Perspective),
                    "centroid" => TokenValue::Sampling(crate::Sampling::Centroid),
                    "sample" => TokenValue::Sampling(crate::Sampling::Sample),
                    "const" => TokenValue::Const,
                    "inout" => TokenValue::InOut,
                    // values
                    "true" => TokenValue::BoolConstant(true),
                    "false" => TokenValue::BoolConstant(false),
                    // jump statements
                    "continue" => TokenValue::Continue,
                    "break" => TokenValue::Break,
                    "return" => TokenValue::Return,
                    "discard" => TokenValue::Discard,
                    // selection statements
                    "if" => TokenValue::If,
                    "else" => TokenValue::Else,
                    "switch" => TokenValue::Switch,
                    "case" => TokenValue::Case,
                    "default" => TokenValue::Default,
                    // iteration statements
                    "while" => TokenValue::While,
                    "do" => TokenValue::Do,
                    "for" => TokenValue::For,
                    // types
                    "void" => TokenValue::Void,
                    "struct" => TokenValue::Struct,
                    word => match parse_type(word) {
                        Some(t) => TokenValue::TypeName(t),
                        None => TokenValue::Identifier(String::from(word)),
                    },
                }
            }
            PPTokenValue::Integer(integer) => TokenValue::IntConstant(integer),
            PPTokenValue::Punct(punct) => match punct {
                // Compound assignments
                Punct::AddAssign => TokenValue::AddAssign,
                Punct::SubAssign => TokenValue::SubAssign,
                Punct::MulAssign => TokenValue::MulAssign,
                Punct::DivAssign => TokenValue::DivAssign,
                Punct::ModAssign => TokenValue::ModAssign,
                Punct::LeftShiftAssign => TokenValue::LeftShiftAssign,
                Punct::RightShiftAssign => TokenValue::RightShiftAssign,
                Punct::AndAssign => TokenValue::AndAssign,
                Punct::XorAssign => TokenValue::XorAssign,
                Punct::OrAssign => TokenValue::OrAssign,

                // Two character punctuation
                Punct::Increment => TokenValue::Increment,
                Punct::Decrement => TokenValue::Decrement,
                Punct::LogicalAnd => TokenValue::LogicalAnd,
                Punct::LogicalOr => TokenValue::LogicalOr,
                Punct::LogicalXor => TokenValue::LogicalXor,
                Punct::LessEqual => TokenValue::LessEqual,
                Punct::GreaterEqual => TokenValue::GreaterEqual,
                Punct::EqualEqual => TokenValue::Equal,
                Punct::NotEqual => TokenValue::NotEqual,
                Punct::LeftShift => TokenValue::LeftShift,
                Punct::RightShift => TokenValue::RightShift,

                // Parenthesis or similar
                Punct::LeftBrace => TokenValue::LeftBrace,
                Punct::RightBrace => TokenValue::RightBrace,
                Punct::LeftParen => TokenValue::LeftParen,
                Punct::RightParen => TokenValue::RightParen,
                Punct::LeftBracket => TokenValue::LeftBracket,
                Punct::RightBracket => TokenValue::RightBracket,

                // Other one character punctuation
                Punct::LeftAngle => TokenValue::LeftAngle,
                Punct::RightAngle => TokenValue::RightAngle,
                Punct::Semicolon => TokenValue::Semicolon,
                Punct::Comma => TokenValue::Comma,
                Punct::Colon => TokenValue::Colon,
                Punct::Dot => TokenValue::Dot,
                Punct::Equal => TokenValue::Assign,
                Punct::Bang => TokenValue::Bang,
                Punct::Minus => TokenValue::Dash,
                Punct::Tilde => TokenValue::Tilde,
                Punct::Plus => TokenValue::Plus,
                Punct::Star => TokenValue::Star,
                Punct::Slash => TokenValue::Slash,
                Punct::Percent => TokenValue::Percent,
                Punct::Pipe => TokenValue::VerticalBar,
                Punct::Caret => TokenValue::Caret,
                Punct::Ampersand => TokenValue::Ampersand,
                Punct::Question => TokenValue::Question,
            },
            PPTokenValue::Pragma(pragma) => {
                for t in pragma.tokens {
                    self.tokens.push_back(t);
                }
                TokenValue::Pragma
            }
            PPTokenValue::Version(version) => {
                for t in version.tokens {
                    self.tokens.push_back(t);
                }
                TokenValue::Version
            }
        };

        Some(Token { value, meta })
    }
}

#[cfg(test)]
mod tests {
    use pp_rs::token::Integer;

    use super::{
        super::token::{SourceMetadata, Token, TokenValue},
        Lexer,
    };

    #[test]
    fn lex_tokens() {
        let defines = crate::FastHashMap::default();

        // line comments
        let mut lex = Lexer::new("#version 450\nvoid main () {}", &defines);
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::Version,
                meta: SourceMetadata { start: 1, end: 8 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::IntConstant(Integer {
                    signed: true,
                    value: 450,
                    width: 32
                }),
                meta: SourceMetadata { start: 9, end: 12 },
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::Void,
                meta: SourceMetadata { start: 13, end: 17 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::Identifier("main".into()),
                meta: SourceMetadata { start: 18, end: 22 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::LeftParen,
                meta: SourceMetadata { start: 23, end: 24 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::RightParen,
                meta: SourceMetadata { start: 24, end: 25 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::LeftBrace,
                meta: SourceMetadata { start: 26, end: 27 }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::RightBrace,
                meta: SourceMetadata { start: 27, end: 28 }
            }
        );
        assert_eq!(lex.next(), None);
    }
}
