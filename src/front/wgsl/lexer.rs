use super::{conv, Error, Token};

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
                (Token::ShiftOperation(cur), chars.as_str())
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
        let word = self.next_ident()?;
        let pair = conv::get_scalar_type(word).ok_or(Error::UnknownScalarType(word))?;
        self.expect(Token::Paren('>'))?;
        Ok(pair)
    }

    pub(super) fn next_format_generic(&mut self) -> Result<crate::StorageFormat, Error<'a>> {
        self.expect(Token::Paren('<'))?;
        let format = conv::map_storage_format(self.next_ident()?)?;
        self.expect(Token::Paren('>'))?;
        Ok(format)
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

#[cfg(test)]
fn sub_test(source: &str, expected_tokens: &[Token]) {
    let mut lex = Lexer::new(source);
    for &token in expected_tokens {
        assert_eq!(lex.next(), token);
    }
    assert_eq!(lex.next(), Token::End);
}

#[test]
fn test_tokens() {
    sub_test("id123_OK", &[Token::Word("id123_OK")]);
    sub_test("92No", &[Token::Number("92"), Token::Word("No")]);
    sub_test(
        "æNoø",
        &[Token::Unknown('æ'), Token::Word("No"), Token::Unknown('ø')],
    );
    sub_test("No¾", &[Token::Word("No"), Token::Unknown('¾')]);
    sub_test("No好", &[Token::Word("No"), Token::Unknown('好')]);
    sub_test("\"\u{2}ПЀ\u{0}\"", &[Token::String("\u{2}ПЀ\u{0}")]); // https://github.com/gfx-rs/naga/issues/90
}

#[test]
fn test_variable_decl() {
    sub_test(
        "[[ group(0 )]] var< uniform> texture:   texture_multisampled_2d <f32 >;",
        &[
            Token::DoubleParen('['),
            Token::Word("group"),
            Token::Paren('('),
            Token::Number("0"),
            Token::Paren(')'),
            Token::DoubleParen(']'),
            Token::Word("var"),
            Token::Paren('<'),
            Token::Word("uniform"),
            Token::Paren('>'),
            Token::Word("texture"),
            Token::Separator(':'),
            Token::Word("texture_multisampled_2d"),
            Token::Paren('<'),
            Token::Word("f32"),
            Token::Paren('>'),
            Token::Separator(';'),
        ],
    )
}
