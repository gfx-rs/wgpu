use super::{
    token::{Token, TokenMetadata, TokenValue},
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
        let mut meta = TokenMetadata {
            line: 0,
            chars: 0..0,
        };
        let pp_token = match self.tokens.pop_front() {
            Some(t) => t,
            None => match self.pp.next()? {
                Ok(t) => t,
                Err((err, loc)) => {
                    meta.line = loc.line as usize;
                    meta.chars.start = loc.pos as usize;
                    //TODO: proper location end
                    meta.chars.end = loc.pos as usize + 1;
                    return Some(Token {
                        value: TokenValue::Unknown(err),
                        meta,
                    });
                }
            },
        };

        meta.line = pp_token.location.line as usize;
        meta.chars.start = pp_token.location.pos as usize;
        //TODO: proper location end
        meta.chars.end = pp_token.location.pos as usize + 1;
        let value = match pp_token.value {
            PPTokenValue::Extension(extension) => {
                for t in extension.tokens {
                    self.tokens.push_back(t);
                }
                TokenValue::Extension
            }
            PPTokenValue::Float(float) => TokenValue::FloatConstant(float.value),
            PPTokenValue::Ident(ident) => {
                match ident.as_str() {
                    "layout" => TokenValue::Layout,
                    "in" => TokenValue::In,
                    "out" => TokenValue::Out,
                    "uniform" => TokenValue::Uniform,
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
            //TODO: unsigned etc
            PPTokenValue::Integer(integer) => TokenValue::IntConstant(integer.value as i64),
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
    use super::{
        super::token::{Token, TokenMetadata, TokenValue},
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
                meta: TokenMetadata {
                    line: 1,
                    chars: 1..2 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::IntConstant(450),
                meta: TokenMetadata {
                    line: 1,
                    chars: 9..10 //TODO
                },
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::Void,
                meta: TokenMetadata {
                    line: 2,
                    chars: 0..1 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::Identifier("main".into()),
                meta: TokenMetadata {
                    line: 2,
                    chars: 5..6 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::LeftParen,
                meta: TokenMetadata {
                    line: 2,
                    chars: 10..11 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::RightParen,
                meta: TokenMetadata {
                    line: 2,
                    chars: 11..12 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::LeftBrace,
                meta: TokenMetadata {
                    line: 2,
                    chars: 13..14 //TODO
                }
            }
        );
        assert_eq!(
            lex.next().unwrap(),
            Token {
                value: TokenValue::RightBrace,
                meta: TokenMetadata {
                    line: 2,
                    chars: 14..15 //TODO
                }
            }
        );
        assert_eq!(lex.next(), None);
    }
}
