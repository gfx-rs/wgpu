use super::parser::Token;
use super::token::TokenMetadata;
use std::{iter::Enumerate, str::Lines};

fn _consume_str<'a>(input: &'a str, what: &str) -> Option<&'a str> {
    if input.starts_with(what) {
        Some(&input[what.len()..])
    } else {
        None
    }
}

fn consume_any(input: &str, what: impl Fn(char) -> bool) -> (&str, &str, usize) {
    let pos = input.find(|c| !what(c)).unwrap_or_else(|| input.len());
    let (o, i) = input.split_at(pos);
    (o, i, pos)
}

pub fn consume_token(mut input: &str) -> (Option<Token>, &str) {
    let start = input
        .find(|c: char| !c.is_whitespace())
        .unwrap_or_else(|| input.chars().count());
    input = &input[start..];

    let mut chars = input.chars();
    let cur = match chars.next() {
        Some(c) => c,
        None => return (None, input),
    };
    let mut meta = TokenMetadata {
        line: 0,
        chars: start..start + 1,
    };
    match cur {
        ':' => (Some(Token::Colon(meta)), chars.as_str()),
        ';' => (Some(Token::Semicolon(meta)), chars.as_str()),
        ',' => (Some(Token::Comma(meta)), chars.as_str()),
        '.' => (Some(Token::Dot(meta)), chars.as_str()),

        '(' => (Some(Token::LeftParen(meta)), chars.as_str()),
        ')' => (Some(Token::RightParen(meta)), chars.as_str()),
        '{' => (Some(Token::LeftBrace(meta)), chars.as_str()),
        '}' => (Some(Token::RightBrace(meta)), chars.as_str()),
        '[' => (Some(Token::LeftBracket(meta)), chars.as_str()),
        ']' => (Some(Token::RightBracket(meta)), chars.as_str()),
        '<' | '>' => {
            input = chars.as_str();
            let n1 = chars.next();
            let input1 = chars.as_str();
            let n2 = chars.next();
            match (cur, n1, n2) {
                ('<', Some('<'), Some('=')) => {
                    meta.chars.end = start + 3;
                    (Some(Token::LeftAssign(meta)), chars.as_str())
                }
                ('>', Some('>'), Some('=')) => {
                    meta.chars.end = start + 3;
                    (Some(Token::RightAssign(meta)), chars.as_str())
                }
                ('<', Some('<'), _) => {
                    meta.chars.end = start + 2;
                    (Some(Token::LeftOp(meta)), input1)
                }
                ('>', Some('>'), _) => {
                    meta.chars.end = start + 2;
                    (Some(Token::RightOp(meta)), input1)
                }
                ('<', Some('='), _) => {
                    meta.chars.end = start + 2;
                    (Some(Token::LeOp(meta)), input1)
                }
                ('>', Some('='), _) => {
                    meta.chars.end = start + 2;
                    (Some(Token::GeOp(meta)), input1)
                }
                ('<', _, _) => (Some(Token::LeftAngle(meta)), input),
                ('>', _, _) => (Some(Token::RightAngle(meta)), input),
                _ => (None, input),
            }
        }
        '0'..='9' => {
            let (number, rest, pos) = consume_any(input, |c| (c >= '0' && c <= '9' || c == '.'));
            if number.find('.').is_some() {
                if (
                    chars.next().map(|c| c.to_lowercase().next().unwrap()),
                    chars.next().map(|c| c.to_lowercase().next().unwrap()),
                ) == (Some('l'), Some('f'))
                {
                    meta.chars.end = start + pos + 2;
                    (
                        Some(Token::DoubleConstant((meta, number.parse().unwrap()))),
                        chars.as_str(),
                    )
                } else {
                    meta.chars.end = start + pos;
                    (
                        Some(Token::FloatConstant((meta, number.parse().unwrap()))),
                        chars.as_str(),
                    )
                }
            } else {
                meta.chars.end = start + pos;
                (
                    Some(Token::IntConstant((meta, number.parse().unwrap()))),
                    rest,
                )
            }
        }
        'a'..='z' | 'A'..='Z' | '_' => {
            let (word, rest, pos) = consume_any(input, |c| c.is_alphanumeric() || c == '_');
            meta.chars.end = start + pos;
            match word {
                "void" => (Some(Token::Void(meta)), rest),
                "vec4" => (Some(Token::Vec4(meta)), rest),
                //TODO: remaining types
                _ => (Some(Token::Identifier((meta, String::from(word)))), rest),
            }
        }

        '+' | '-' | '&' | '|' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    match cur {
                        '+' => (Some(Token::AddAssign(meta)), chars.as_str()),
                        '-' => (Some(Token::SubAssign(meta)), chars.as_str()),
                        '&' => (Some(Token::AndAssign(meta)), chars.as_str()),
                        '|' => (Some(Token::OrAssign(meta)), chars.as_str()),
                        '^' => (Some(Token::XorAssign(meta)), chars.as_str()),
                        _ => (None, input),
                    }
                }
                Some(cur) => {
                    meta.chars.end = start + 2;
                    match cur {
                        '+' => (Some(Token::IncOp(meta)), chars.as_str()),
                        '-' => (Some(Token::DecOp(meta)), chars.as_str()),
                        '&' => (Some(Token::AndOp(meta)), chars.as_str()),
                        '|' => (Some(Token::OrOp(meta)), chars.as_str()),
                        '^' => (Some(Token::XorOp(meta)), chars.as_str()),
                        _ => (None, input),
                    }
                }
                _ => match cur {
                    '+' => (Some(Token::Plus(meta)), input),
                    '-' => (Some(Token::Dash(meta)), input),
                    '&' => (Some(Token::Ampersand(meta)), input),
                    '|' => (Some(Token::VerticalBar(meta)), input),
                    '^' => (Some(Token::Caret(meta)), input),
                    _ => (None, input),
                },
            }
        }

        '%' | '!' | '=' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    match cur {
                        '%' => (Some(Token::ModAssign(meta)), chars.as_str()),
                        '!' => (Some(Token::NeOp(meta)), chars.as_str()),
                        '=' => (Some(Token::EqOp(meta)), chars.as_str()),
                        _ => (None, input),
                    }
                }
                _ => match cur {
                    '%' => (Some(Token::Percent(meta)), input),
                    '!' => (Some(Token::Bang(meta)), input),
                    '=' => (Some(Token::Equal(meta)), input),
                    _ => (None, input),
                },
            }
        }

        '*' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    (Some(Token::MulAssign(meta)), chars.as_str())
                }
                //TODO: multi-line comments
                // Some('/') => (
                //     Token::MultiLineCommentClose,
                //     chars.as_str(),
                //     start,
                //     start + 2,
                // ),
                _ => (Some(Token::MulAssign(meta)), input),
            }
        }
        '/' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    (Some(Token::DivAssign(meta)), chars.as_str())
                }
                Some('/') => (None, ""),
                //TODO: multi-line comments
                // Some('*') => (
                //     Token::MultiLineCommentOpen,
                //     chars.as_str(),
                //     start,
                //     start + 2,
                // ),
                _ => (Some(Token::Slash(meta)), input),
            }
        }
        '#' => {
            input = chars.as_str();
            let (word, rest, pos) = consume_any(input, |c| c.is_alphanumeric() || c == '_');
            meta.chars.end = start + pos;
            match word {
                "version" => (Some(Token::Version(meta)), rest),
                _ => (None, input),
            }

            //TODO: preprocessor
            // if chars.next() == Some(cur) {
            //     (Token::TokenPasting, chars.as_str(), start, start + 2)
            // } else {
            //     (Token::Preprocessor, input, start, start + 1)
            // }
        }
        '~' => (Some(Token::Tilde(meta)), chars.as_str()),
        '?' => (Some(Token::Question(meta)), chars.as_str()),
        _ => (None, chars.as_str()),
    }
}

#[derive(Clone, Debug)]
pub struct Lexer<'a> {
    lines: Enumerate<Lines<'a>>,
    input: String,
    line: usize,
    offset: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lines = input.lines().enumerate();
        let (line, input) = lines.next().unwrap_or((0, ""));
        let mut input = String::from(input);

        while input.ends_with('\\') {
            if let Some((_, next)) = lines.next() {
                input.pop();
                input.push_str(next);
            } else {
                break;
            }
        }

        Lexer {
            lines,
            input,
            line,
            offset: 0,
        }
    }

    #[must_use]
    pub fn next(&mut self) -> Option<Token> {
        let (token, rest) = consume_token(&self.input);

        if let Some(mut token) = token {
            self.input = String::from(rest);
            let meta = token.extra_mut();
            let end = meta.chars.end;
            meta.line = self.line;
            meta.chars.start += self.offset;
            meta.chars.end += self.offset;
            self.offset += end;
            Some(token)
        } else {
            let (line, input) = self.lines.next()?;

            let mut input = String::from(input);

            while input.ends_with('\\') {
                if let Some((_, next)) = self.lines.next() {
                    input.pop();
                    input.push_str(next);
                } else {
                    break;
                }
            }

            self.input = input;
            self.line = line;
            self.offset = 0;
            self.next()
        }
    }

    // #[must_use]
    // pub fn peek(&mut self) -> Option<Token> {
    //     self.clone().next()
    // }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}
