use super::parser::Token;
use super::{preprocess::LinePreProcessor, token::TokenMetadata, types::parse_type};
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

#[derive(Clone, Debug)]
pub struct Lexer<'a> {
    lines: Enumerate<Lines<'a>>,
    input: String,
    line: usize,
    offset: usize,
    inside_comment: bool,
    pub pp: LinePreProcessor,
}

impl<'a> Lexer<'a> {
    pub fn consume_token(&mut self) -> Option<Token> {
        let start = self
            .input
            .find(|c: char| !c.is_whitespace())
            .unwrap_or_else(|| self.input.chars().count());
        let input = &self.input[start..];

        let mut chars = input.chars();
        let cur = match chars.next() {
            Some(c) => c,
            None => {
                self.input = self.input[start..].into();
                return None;
            }
        };
        let mut meta = TokenMetadata {
            line: 0,
            chars: start..start + 1,
        };
        let mut consume_all = false;
        let token = match cur {
            ':' => Some(Token::Colon(meta)),
            ';' => Some(Token::Semicolon(meta)),
            ',' => Some(Token::Comma(meta)),
            '.' => Some(Token::Dot(meta)),

            '(' => Some(Token::LeftParen(meta)),
            ')' => Some(Token::RightParen(meta)),
            '{' => Some(Token::LeftBrace(meta)),
            '}' => Some(Token::RightBrace(meta)),
            '[' => Some(Token::LeftBracket(meta)),
            ']' => Some(Token::RightBracket(meta)),
            '<' | '>' => {
                let n1 = chars.next();
                let n2 = chars.next();
                match (cur, n1, n2) {
                    ('<', Some('<'), Some('=')) => {
                        meta.chars.end = start + 3;
                        Some(Token::LeftAssign(meta))
                    }
                    ('>', Some('>'), Some('=')) => {
                        meta.chars.end = start + 3;
                        Some(Token::RightAssign(meta))
                    }
                    ('<', Some('<'), _) => {
                        meta.chars.end = start + 2;
                        Some(Token::LeftOp(meta))
                    }
                    ('>', Some('>'), _) => {
                        meta.chars.end = start + 2;
                        Some(Token::RightOp(meta))
                    }
                    ('<', Some('='), _) => {
                        meta.chars.end = start + 2;
                        Some(Token::LeOp(meta))
                    }
                    ('>', Some('='), _) => {
                        meta.chars.end = start + 2;
                        Some(Token::GeOp(meta))
                    }
                    ('<', _, _) => Some(Token::LeftAngle(meta)),
                    ('>', _, _) => Some(Token::RightAngle(meta)),
                    _ => None,
                }
            }
            '0'..='9' => {
                let (number, _, pos) = consume_any(input, |c| (c >= '0' && c <= '9' || c == '.'));
                if number.find('.').is_some() {
                    if (
                        chars.next().map(|c| c.to_lowercase().next().unwrap()),
                        chars.next().map(|c| c.to_lowercase().next().unwrap()),
                    ) == (Some('l'), Some('f'))
                    {
                        meta.chars.end = start + pos + 2;
                        Some(Token::DoubleConstant((meta, number.parse().unwrap())))
                    } else {
                        meta.chars.end = start + pos;

                        Some(Token::FloatConstant((meta, number.parse().unwrap())))
                    }
                } else {
                    meta.chars.end = start + pos;
                    Some(Token::IntConstant((meta, number.parse().unwrap())))
                }
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let (word, _, pos) = consume_any(input, |c| c.is_ascii_alphanumeric() || c == '_');
                meta.chars.end = start + pos;
                match word {
                    "layout" => Some(Token::Layout(meta)),
                    "in" => Some(Token::In(meta)),
                    "out" => Some(Token::Out(meta)),
                    "uniform" => Some(Token::Uniform(meta)),
                    "flat" => Some(Token::Interpolation((meta, crate::Interpolation::Flat))),
                    "noperspective" => {
                        Some(Token::Interpolation((meta, crate::Interpolation::Linear)))
                    }
                    "smooth" => Some(Token::Interpolation((
                        meta,
                        crate::Interpolation::Perspective,
                    ))),
                    "centroid" => {
                        Some(Token::Interpolation((meta, crate::Interpolation::Centroid)))
                    }
                    "sample" => Some(Token::Interpolation((meta, crate::Interpolation::Sample))),
                    // values
                    "true" => Some(Token::BoolConstant((meta, true))),
                    "false" => Some(Token::BoolConstant((meta, false))),
                    // jump statements
                    "continue" => Some(Token::Continue(meta)),
                    "break" => Some(Token::Break(meta)),
                    "return" => Some(Token::Return(meta)),
                    "discard" => Some(Token::Discard(meta)),
                    // selection statements
                    "if" => Some(Token::If(meta)),
                    "else" => Some(Token::Else(meta)),
                    "switch" => Some(Token::Switch(meta)),
                    "case" => Some(Token::Case(meta)),
                    "default" => Some(Token::Default(meta)),
                    // iteration statements
                    "while" => Some(Token::While(meta)),
                    "do" => Some(Token::Do(meta)),
                    "for" => Some(Token::For(meta)),
                    // types
                    "void" => Some(Token::Void(meta)),
                    word => {
                        let token = match parse_type(word) {
                            Some(t) => Token::TypeName((meta, t)),
                            None => Token::Identifier((meta, String::from(word))),
                        };
                        Some(token)
                    }
                }
            }
            '+' | '-' | '&' | '|' | '^' => {
                let next = chars.next();
                if next == Some(cur) {
                    meta.chars.end = start + 2;
                    match cur {
                        '+' => Some(Token::IncOp(meta)),
                        '-' => Some(Token::DecOp(meta)),
                        '&' => Some(Token::AndOp(meta)),
                        '|' => Some(Token::OrOp(meta)),
                        '^' => Some(Token::XorOp(meta)),
                        _ => None,
                    }
                } else {
                    match next {
                        Some('=') => {
                            meta.chars.end = start + 2;
                            match cur {
                                '+' => Some(Token::AddAssign(meta)),
                                '-' => Some(Token::SubAssign(meta)),
                                '&' => Some(Token::AndAssign(meta)),
                                '|' => Some(Token::OrAssign(meta)),
                                '^' => Some(Token::XorAssign(meta)),
                                _ => None,
                            }
                        }
                        _ => match cur {
                            '+' => Some(Token::Plus(meta)),
                            '-' => Some(Token::Dash(meta)),
                            '&' => Some(Token::Ampersand(meta)),
                            '|' => Some(Token::VerticalBar(meta)),
                            '^' => Some(Token::Caret(meta)),
                            _ => None,
                        },
                    }
                }
            }

            '%' | '!' | '=' => match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    match cur {
                        '%' => Some(Token::ModAssign(meta)),
                        '!' => Some(Token::NeOp(meta)),
                        '=' => Some(Token::EqOp(meta)),
                        _ => None,
                    }
                }
                _ => match cur {
                    '%' => Some(Token::Percent(meta)),
                    '!' => Some(Token::Bang(meta)),
                    '=' => Some(Token::Equal(meta)),
                    _ => None,
                },
            },

            '*' => match chars.next() {
                Some('=') => {
                    meta.chars.end = start + 2;
                    Some(Token::MulAssign(meta))
                }
                Some('/') => {
                    meta.chars.end = start + 2;
                    Some(Token::CommentEnd((meta, ())))
                }
                _ => Some(Token::Star(meta)),
            },
            '/' => {
                match chars.next() {
                    Some('=') => {
                        meta.chars.end = start + 2;
                        Some(Token::DivAssign(meta))
                    }
                    Some('/') => {
                        // consume rest of line
                        consume_all = true;
                        None
                    }
                    Some('*') => {
                        meta.chars.end = start + 2;
                        Some(Token::CommentStart((meta, ())))
                    }
                    _ => Some(Token::Slash(meta)),
                }
            }
            '#' => {
                if self.offset == 0 {
                    let mut input = chars.as_str();

                    // skip whitespace
                    let word_start = input
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or_else(|| input.chars().count());
                    input = &input[word_start..];

                    let (word, _, pos) = consume_any(input, |c| c.is_alphanumeric() || c == '_');
                    meta.chars.end = start + word_start + 1 + pos;
                    match word {
                        "version" => Some(Token::Version(meta)),
                        w => Some(Token::Unknown((meta, w.into()))),
                    }

                //TODO: preprocessor
                // if chars.next() == Some(cur) {
                //     (Token::TokenPasting, chars.as_str(), start, start + 2)
                // } else {
                //     (Token::Preprocessor, input, start, start + 1)
                // }
                } else {
                    Some(Token::Unknown((meta, '#'.to_string())))
                }
            }
            '~' => Some(Token::Tilde(meta)),
            '?' => Some(Token::Question(meta)),
            ch => Some(Token::Unknown((meta, ch.to_string()))),
        };
        if let Some(token) = token {
            let skip_bytes = input
                .chars()
                .take(token.extra().chars.end - start)
                .fold(0, |acc, c| acc + c.len_utf8());
            self.input = input[skip_bytes..].into();
            Some(token)
        } else {
            if consume_all {
                self.input = "".into();
            } else {
                self.input = self.input[start..].into();
            }
            None
        }
    }

    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            lines: input.lines().enumerate(),
            input: "".to_string(),
            line: 0,
            offset: 0,
            inside_comment: false,
            pp: LinePreProcessor::new(),
        };
        lexer.next_line();
        lexer
    }

    fn next_line(&mut self) -> bool {
        if let Some((line, input)) = self.lines.next() {
            let mut input = String::from(input);

            while input.ends_with('\\') {
                if let Some((_, next)) = self.lines.next() {
                    input.pop();
                    input.push_str(next);
                } else {
                    break;
                }
            }

            if let Ok(processed) = self.pp.process_line(&input) {
                self.input = processed.unwrap_or_default();
                self.line = line;
                self.offset = 0;
                true
            } else {
                //TODO: handle preprocessor error
                false
            }
        } else {
            false
        }
    }

    #[must_use]
    pub fn next(&mut self) -> Option<Token> {
        let token = self.consume_token();

        if let Some(mut token) = token {
            let meta = token.extra_mut();
            let end = meta.chars.end;
            meta.line = self.line;
            meta.chars.start += self.offset;
            meta.chars.end += self.offset;
            self.offset += end;
            if !self.inside_comment {
                match token {
                    Token::CommentStart(_) => {
                        self.inside_comment = true;
                        self.next()
                    }
                    _ => Some(token),
                }
            } else {
                if let Token::CommentEnd(_) = token {
                    self.inside_comment = false;
                }
                self.next()
            }
        } else {
            if !self.next_line() {
                return None;
            }
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
