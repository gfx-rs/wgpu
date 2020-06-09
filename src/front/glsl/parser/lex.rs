use super::{Token, TokenMetadata};
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

pub fn consume_token(input: &String) -> (Token, &str, usize, usize) {
    let mut input = input.as_str();

    let start = input
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(input.chars().count());
    input = &input[start..];

    let mut chars = input.chars();
    let cur = match chars.next() {
        Some(c) => c,
        None => return (Token::End, input, start, start + 1),
    };
    match cur {
        ':' => {
            input = chars.as_str();
            if chars.next() == Some(':') {
                (Token::DoubleColon, chars.as_str(), start, start + 2)
            } else {
                (Token::Separator(cur), input, start, start + 1)
            }
        }
        ';' | ',' | '.' => (Token::Separator(cur), chars.as_str(), start, start + 1),
        '(' | ')' | '{' | '}' | '[' | ']' => (Token::Paren(cur), chars.as_str(), start, start + 1),
        '<' | '>' => {
            input = chars.as_str();
            let next = chars.next();
            if next == Some('=') {
                (
                    Token::LogicalOperation(cur),
                    chars.as_str(),
                    start,
                    start + 1,
                )
            } else if next == Some(cur) {
                (Token::ShiftOperation(cur), chars.as_str(), start, start + 2)
            } else {
                (Token::Operation(cur), input, start, start + 1)
            }
        }
        '0'..='9' => {
            let (number, rest, pos) = consume_any(input, |c| (c >= '0' && c <= '9' || c == '.'));
            if let Some(_) = number.find('.') {
                if (
                    chars.next().map(|c| c.to_lowercase().next().unwrap()),
                    chars.next().map(|c| c.to_lowercase().next().unwrap()),
                ) == (Some('l'), Some('f'))
                {
                    (
                        Token::Double(number.parse().unwrap()),
                        chars.as_str(),
                        start,
                        start + pos + 2,
                    )
                } else {
                    (
                        Token::Float(number.parse().unwrap()),
                        chars.as_str(),
                        start,
                        start + pos,
                    )
                }
            } else {
                (
                    Token::Integral(number.parse().unwrap()),
                    rest,
                    start,
                    start + pos,
                )
            }
        }
        'a'..='z' | 'A'..='Z' | '_' => {
            let (word, rest, pos) = consume_any(input, |c| c.is_alphanumeric() || c == '_');
            (Token::Word(String::from(word)), rest, start, start + pos)
        }
        '+' | '-' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => (Token::OpAssign(cur), chars.as_str(), start, start + 2),
                Some(next) if cur == next => (Token::Sufix(cur), chars.as_str(), start, start + 2),
                _ => (Token::Operation(cur), input, start, start + 1),
            }
        }
        '%' | '^' => {
            input = chars.as_str();

            if chars.next() == Some('=') {
                (Token::OpAssign(cur), chars.as_str(), start, start + 2)
            } else {
                (Token::Operation(cur), input, start, start + 1)
            }
        }
        '!' => {
            input = chars.as_str();

            if chars.next() == Some('=') {
                (
                    Token::LogicalOperation(cur),
                    chars.as_str(),
                    start,
                    start + 2,
                )
            } else {
                (Token::Operation(cur), input, start, start + 1)
            }
        }
        '*' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => (Token::OpAssign(cur), chars.as_str(), start, start + 2),
                Some('/') => (
                    Token::MultiLineCommentClose,
                    chars.as_str(),
                    start,
                    start + 2,
                ),
                _ => (Token::Operation(cur), input, start, start + 1),
            }
        }
        '/' => {
            input = chars.as_str();
            match chars.next() {
                Some('=') => (Token::OpAssign(cur), chars.as_str(), start, start + 2),
                Some('/') => (Token::LineComment, chars.as_str(), start, start + 2),
                Some('*') => (
                    Token::MultiLineCommentOpen,
                    chars.as_str(),
                    start,
                    start + 2,
                ),
                _ => (Token::Operation(cur), input, start, start + 1),
            }
        }
        '=' | '&' | '|' => {
            input = chars.as_str();
            if chars.next() == Some(cur) {
                (
                    Token::LogicalOperation(cur),
                    chars.as_str(),
                    start,
                    start + 2,
                )
            } else {
                (Token::Operation(cur), input, start, start + 1)
            }
        }
        '#' => {
            input = chars.as_str();
            if chars.next() == Some(cur) {
                (Token::TokenPasting, chars.as_str(), start, start + 2)
            } else {
                (Token::Preprocessor, input, start, start + 1)
            }
        }
        '~' => (Token::Operation(cur), chars.as_str(), start, start + 1),
        '?' => (Token::Selection, chars.as_str(), start, start + 1),
        _ => (Token::Unknown(cur), chars.as_str(), start, start + 1),
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

        while input.chars().last() == Some('\\') {
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
    pub fn next(&mut self) -> TokenMetadata {
        let (token, rest, start, end) = consume_token(&self.input);

        if token == Token::End {
            match self.lines.next() {
                Some((line, input)) => {
                    let mut input = String::from(input);

                    while input.chars().last() == Some('\\') {
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
                None => TokenMetadata {
                    token: Token::End,
                    line: self.line,
                    chars: self.offset + start..end + self.offset,
                },
            }
        } else {
            self.input = String::from(rest);
            let metadata = TokenMetadata {
                token,
                line: self.line,
                chars: self.offset + start..end + self.offset,
            };
            self.offset += end;
            metadata
        }
    }

    #[must_use]
    pub fn peek(&mut self) -> TokenMetadata {
        self.clone().next()
    }
}
