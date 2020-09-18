use super::{Error, Token};

fn _consume_str<'a>(input: &'a str, what: &str) -> Option<&'a str> {
    if input.starts_with(what) {
        Some(&input[what.len()..])
    } else {
        None
    }
}

fn consume_any(input: &str, what: impl Fn(char) -> bool) -> (&str, &str) {
    let pos = input.find(|c| !what(c)).unwrap_or_else(|| input.len());
    input.split_at(pos)
}

fn consume_number(input: &str) -> (&str, &str) {
    let mut is_first_char = true;
    let mut right_after_exponent = false;

    let mut what = |c| {
        if is_first_char {
            is_first_char = false;
            c == '-' || c >= '0' && c <= '9' || c == '.'
        } else if c == 'e' || c == 'E' {
            right_after_exponent = true;
            true
        } else if right_after_exponent {
            right_after_exponent = false;
            c >= '0' && c <= '9' || c == '-'
        } else {
            c >= '0' && c <= '9' || c == '.'
        }
    };
    let pos = input.find(|c| !what(c)).unwrap_or_else(|| input.len());
    input.split_at(pos)
}

fn consume_token(mut input: &str) -> (Token<'_>, &str) {
    input = input.trim_start();
    let mut chars = input.chars();
    let cur = match chars.next() {
        Some(c) => c,
        None => return (Token::End, input),
    };
    match cur {
        ':' => {
            input = chars.as_str();
            if chars.next() == Some(':') {
                (Token::DoubleColon, chars.as_str())
            } else {
                (Token::Separator(cur), input)
            }
        }
        ';' | ',' => (Token::Separator(cur), chars.as_str()),
        '.' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('0'..='9') => {
                    let (number, rest) = consume_number(input);
                    (Token::Number(number), rest)
                }
                _ => (Token::Separator(cur), og_chars),
            }
        }
        '(' | ')' | '{' | '}' => (Token::Paren(cur), chars.as_str()),
        '<' | '>' => {
            input = chars.as_str();
            let next = chars.next();
            if next == Some('=') {
                (Token::LogicalOperation(cur), chars.as_str())
            } else if next == Some(cur) {
                input = chars.as_str();
                if chars.next() == Some(cur) {
                    (Token::ArithmeticShiftOperation(cur), chars.as_str())
                } else {
                    (Token::ShiftOperation(cur), input)
                }
            } else {
                (Token::Paren(cur), input)
            }
        }
        '[' | ']' => {
            input = chars.as_str();
            if chars.next() == Some(cur) {
                (Token::DoubleParen(cur), chars.as_str())
            } else {
                (Token::Paren(cur), input)
            }
        }
        '0'..='9' => {
            let (number, rest) = consume_number(input);
            (Token::Number(number), rest)
        }
        'a'..='z' | 'A'..='Z' | '_' => {
            let (word, rest) = consume_any(input, |c| c.is_ascii_alphanumeric() || c == '_');
            (Token::Word(word), rest)
        }
        '"' => {
            let mut iter = chars.as_str().splitn(2, '"');

            // splitn returns an iterator with at least one element, so unwrapping is fine
            let quote_content = iter.next().unwrap();
            if let Some(rest) = iter.next() {
                (Token::String(quote_content), rest)
            } else {
                (Token::UnterminatedString, quote_content)
            }
        }
        '-' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('>') => (Token::Arrow, chars.as_str()),
                Some('0'..='9') | Some('.') => {
                    let (number, rest) = consume_number(input);
                    (Token::Number(number), rest)
                }
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '+' | '*' | '/' | '%' | '^' => (Token::Operation(cur), chars.as_str()),
        '!' => {
            if chars.next() == Some('=') {
                (Token::LogicalOperation(cur), chars.as_str())
            } else {
                (Token::Operation(cur), input)
            }
        }
        '=' | '&' | '|' => {
            input = chars.as_str();
            if chars.next() == Some(cur) {
                (Token::LogicalOperation(cur), chars.as_str())
            } else {
                (Token::Operation(cur), input)
            }
        }
        '#' => match chars.position(|c| c == '\n' || c == '\r') {
            Some(_) => consume_token(chars.as_str()),
            None => (Token::End, chars.as_str()),
        },
        _ => (Token::Unknown(cur), chars.as_str()),
    }
}

#[derive(Clone)]
pub(super) struct Lexer<'a> {
    input: &'a str,
}

impl<'a> Lexer<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        Lexer { input }
    }

    #[must_use]
    pub(super) fn next(&mut self) -> Token<'a> {
        let (token, rest) = consume_token(self.input);
        self.input = rest;
        token
    }

    #[must_use]
    pub(super) fn peek(&mut self) -> Token<'a> {
        self.clone().next()
    }

    pub(super) fn expect(&mut self, expected: Token<'_>) -> Result<(), Error<'a>> {
        let token = self.next();
        if token == expected {
            Ok(())
        } else {
            Err(Error::Unexpected(token))
        }
    }

    pub(super) fn skip(&mut self, what: Token<'_>) -> bool {
        let (token, rest) = consume_token(self.input);
        if token == what {
            self.input = rest;
            true
        } else {
            false
        }
    }

    pub(super) fn next_ident(&mut self) -> Result<&'a str, Error<'a>> {
        match self.next() {
            Token::Word(word) => Ok(word),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _next_float_literal(&mut self) -> Result<f32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadFloat(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    pub(super) fn next_uint_literal(&mut self) -> Result<u32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    fn _next_sint_literal(&mut self) -> Result<i32, Error<'a>> {
        match self.next() {
            Token::Number(word) => word.parse().map_err(|err| Error::BadInteger(word, err)),
            other => Err(Error::Unexpected(other)),
        }
    }

    pub(super) fn next_scalar_generic(
        &mut self,
    ) -> Result<(crate::ScalarKind, crate::Bytes), Error<'a>> {
        self.expect(Token::Paren('<'))?;
        let pair = match self.next() {
            Token::Word("f32") => (crate::ScalarKind::Float, 4),
            Token::Word("i32") => (crate::ScalarKind::Sint, 4),
            Token::Word("u32") => (crate::ScalarKind::Uint, 4),
            other => return Err(Error::Unexpected(other)),
        };
        self.expect(Token::Paren('>'))?;
        Ok(pair)
    }

    pub(super) fn next_format_generic(&mut self) -> Result<crate::StorageFormat, Error<'a>> {
        use crate::StorageFormat as Sf;
        self.expect(Token::Paren('<'))?;
        let pair = match self.next() {
            Token::Word("r8unorm") => Sf::R8Unorm,
            Token::Word("r8snorm") => Sf::R8Snorm,
            Token::Word("r8uint") => Sf::R8Uint,
            Token::Word("r8sint") => Sf::R8Sint,
            Token::Word("r16uint") => Sf::R16Uint,
            Token::Word("r16sint") => Sf::R16Sint,
            Token::Word("r16float") => Sf::R16Float,
            Token::Word("rg8unorm") => Sf::Rg8Unorm,
            Token::Word("rg8snorm") => Sf::Rg8Snorm,
            Token::Word("rg8uint") => Sf::Rg8Uint,
            Token::Word("rg8sint") => Sf::Rg8Sint,
            Token::Word("r32uint") => Sf::R32Uint,
            Token::Word("r32sint") => Sf::R32Sint,
            Token::Word("r32float") => Sf::R32Float,
            Token::Word("rg16uint") => Sf::Rg16Uint,
            Token::Word("rg16sint") => Sf::Rg16Sint,
            Token::Word("rg16float") => Sf::Rg16Float,
            Token::Word("rgba8unorm") => Sf::Rgba8Unorm,
            Token::Word("rgba8snorm") => Sf::Rgba8Snorm,
            Token::Word("rgba8uint") => Sf::Rgba8Uint,
            Token::Word("rgba8sint") => Sf::Rgba8Sint,
            Token::Word("rgb10a2unorm") => Sf::Rgb10a2Unorm,
            Token::Word("rg11b10float") => Sf::Rg11b10Float,
            Token::Word("rg32uint") => Sf::Rg32Uint,
            Token::Word("rg32sint") => Sf::Rg32Sint,
            Token::Word("rg32float") => Sf::Rg32Float,
            Token::Word("rgba16uint") => Sf::Rgba16Uint,
            Token::Word("rgba16sint") => Sf::Rgba16Sint,
            Token::Word("rgba16float") => Sf::Rgba16Float,
            Token::Word("rgba32uint") => Sf::Rgba32Uint,
            Token::Word("rgba32sint") => Sf::Rgba32Sint,
            Token::Word("rgba32float") => Sf::Rgba32Float,
            other => return Err(Error::Unexpected(other)),
        };
        self.expect(Token::Paren('>'))?;
        Ok(pair)
    }

    pub(super) fn take_until(&mut self, what: Token<'_>) -> Result<Lexer<'a>, Error<'a>> {
        let original_input = self.input;
        let initial_len = self.input.len();
        let mut used_len = 0;
        loop {
            if self.next() == what {
                break;
            }
            used_len = initial_len - self.input.len();
        }

        Ok(Lexer {
            input: &original_input[..used_len],
        })
    }

    pub(super) fn offset_from(&self, source: &'a str) -> usize {
        source.len() - self.input.len()
    }
}
