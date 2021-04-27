use super::{conv, Error, Span, Token, TokenSpan};

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

fn consume_number(input: &str) -> (Token, &str) {
    //Note: I wish this function was simpler and faster...
    let mut is_first_char = true;
    let mut right_after_exponent = false;

    let mut what = |c| {
        if is_first_char {
            is_first_char = false;
            c == '-' || ('0'..='9').contains(&c) || c == '.'
        } else if c == 'e' || c == 'E' {
            right_after_exponent = true;
            true
        } else if right_after_exponent {
            right_after_exponent = false;
            ('0'..='9').contains(&c) || c == '-'
        } else {
            ('0'..='9').contains(&c) || c == '.'
        }
    };
    let pos = input.find(|c| !what(c)).unwrap_or_else(|| input.len());
    let (value, rest) = input.split_at(pos);

    let mut rest_iter = rest.chars();
    let ty = rest_iter.next().unwrap_or(' ');
    match ty {
        'u' | 'i' | 'f' => {
            let width_end = rest_iter
                .position(|c| !('0'..='9').contains(&c))
                .unwrap_or_else(|| rest.len() - 1);
            let (width, rest) = rest[1..].split_at(width_end);
            (Token::Number { value, ty, width }, rest)
        }
        // default to `i32` or `f32`
        _ => (
            Token::Number {
                value,
                ty: if value.contains('.') { 'f' } else { 'i' },
                width: "",
            },
            rest,
        ),
    }
}

fn consume_token(mut input: &str, generic: bool) -> (Token<'_>, &str) {
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
                Some('0'..='9') => consume_number(input),
                _ => (Token::Separator(cur), og_chars),
            }
        }
        '(' | ')' | '{' | '}' => (Token::Paren(cur), chars.as_str()),
        '<' | '>' => {
            input = chars.as_str();
            let next = chars.next();
            if next == Some('=') && !generic {
                (Token::LogicalOperation(cur), chars.as_str())
            } else if next == Some(cur) && !generic {
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
        '0'..='9' => consume_number(input),
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
        '/' if chars.as_str().starts_with('/') => {
            let _ = chars.position(|c| c == '\n' || c == '\r');
            (Token::Trivia, chars.as_str())
        }
        '-' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('>') => (Token::Arrow, chars.as_str()),
                Some('0'..='9') | Some('.') => consume_number(input),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '+' | '*' | '/' | '%' | '^' => (Token::Operation(cur), chars.as_str()),
        '!' | '~' => {
            input = chars.as_str();
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
        ' ' | '\n' | '\r' | '\t' => {
            let (_, rest) = consume_any(input, |c| c == ' ' || c == '\n' || c == '\r' || c == '\t');
            (Token::Trivia, rest)
        }
        _ => (Token::Unknown(cur), chars.as_str()),
    }
}

#[derive(Clone)]
pub(super) struct Lexer<'a> {
    input: &'a str,
    pub(super) source: &'a str,
}

impl<'a> Lexer<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        Lexer {
            input,
            source: input,
        }
    }

    pub(super) fn _leftover_span(&self) -> Span {
        self.source.len() - self.input.len()..self.source.len()
    }

    fn peek_token_and_rest(&mut self) -> (TokenSpan<'a>, &'a str) {
        let mut cloned = self.clone();
        let token = cloned.next();
        let rest = cloned.input;
        (token, rest)
    }

    fn current_byte_offset(&self) -> usize {
        self.source.len() - self.input.len()
    }

    #[must_use]
    pub(super) fn next(&mut self) -> TokenSpan<'a> {
        let mut start_byte_offset = self.current_byte_offset();
        loop {
            let (token, rest) = consume_token(self.input, false);
            self.input = rest;
            match token {
                Token::Trivia => start_byte_offset = self.current_byte_offset(),
                _ => return (token, start_byte_offset..self.current_byte_offset()),
            }
        }
    }

    #[must_use]
    pub(super) fn next_generic(&mut self) -> TokenSpan<'a> {
        let mut start_byte_offset = self.current_byte_offset();
        loop {
            let (token, rest) = consume_token(self.input, true);
            self.input = rest;
            match token {
                Token::Trivia => start_byte_offset = self.current_byte_offset(),
                _ => return (token, start_byte_offset..self.current_byte_offset()),
            }
        }
    }

    #[must_use]
    pub(super) fn peek(&mut self) -> TokenSpan<'a> {
        let (token, _) = self.peek_token_and_rest();
        token
    }

    pub(super) fn expect(&mut self, expected: Token<'a>) -> Result<(), Error<'a>> {
        let next = self.next();
        if next.0 == expected {
            Ok(())
        } else {
            let description = match expected {
                Token::Separator(_) => "separator",
                Token::DoubleColon => "::",
                Token::Paren(_) => "paren",
                Token::DoubleParen(_) => "double paren",
                Token::Number { .. } => "number",
                Token::String(string) => string,
                Token::Word(word) => word,
                Token::Operation(_) => "operation",
                Token::LogicalOperation(_) => "logical op",
                Token::ShiftOperation(_) => "shift op",
                Token::Arrow => "->",
                Token::Unknown(_) => "unknown",
                Token::UnterminatedString => "string",
                Token::Trivia => "trivia",
                Token::End => "",
            };
            Err(Error::Unexpected(next, description))
        }
    }

    pub(super) fn expect_generic_paren(&mut self, expected: char) -> Result<(), Error<'a>> {
        let next = self.next_generic();
        if next.0 == Token::Paren(expected) {
            Ok(())
        } else {
            Err(Error::Unexpected(next, "paren"))
        }
    }

    pub(super) fn skip(&mut self, what: Token<'_>) -> bool {
        let (peeked_token, rest) = self.peek_token_and_rest();
        if peeked_token.0 == what {
            self.input = rest;
            true
        } else {
            false
        }
    }

    pub(super) fn next_ident_with_span(&mut self) -> Result<(&'a str, Span), Error<'a>> {
        match self.next() {
            (Token::Word(word), span) => Ok((word, span)),
            other => Err(Error::Unexpected(other, "identifier")),
        }
    }

    pub(super) fn next_ident(&mut self) -> Result<&'a str, Error<'a>> {
        match self.next() {
            (Token::Word(word), _) => Ok(word),
            other => Err(Error::Unexpected(other, "identifier")),
        }
    }

    fn _next_float_literal(&mut self) -> Result<f32, Error<'a>> {
        match self.next() {
            (Token::Number { value, .. }, span) => value.parse().map_err(|_| Error::BadFloat(span)),
            other => Err(Error::Unexpected(other, "floating-point literal")),
        }
    }

    pub(super) fn next_uint_literal(&mut self) -> Result<u32, Error<'a>> {
        match self.next() {
            (Token::Number { value, .. }, span) => {
                value.parse().map_err(|_| Error::BadInteger(span))
            }
            other => Err(Error::Unexpected(other, "unsigned integer literal")),
        }
    }

    pub(super) fn next_sint_literal(&mut self) -> Result<i32, Error<'a>> {
        match self.next() {
            (Token::Number { value, .. }, span) => {
                value.parse().map_err(|_| Error::BadInteger(span))
            }
            other => Err(Error::Unexpected(other, "signed integer literal")),
        }
    }

    pub(super) fn next_scalar_generic(
        &mut self,
    ) -> Result<(crate::ScalarKind, crate::Bytes), Error<'a>> {
        self.expect_generic_paren('<')?;
        let word = self.next_ident()?;
        let pair = conv::get_scalar_type(word).ok_or(Error::UnknownScalarType(word))?;
        self.expect_generic_paren('>')?;
        Ok(pair)
    }

    pub(super) fn next_format_generic(&mut self) -> Result<crate::StorageFormat, Error<'a>> {
        self.expect(Token::Paren('<'))?;
        let format = conv::map_storage_format(self.next_ident()?)?;
        self.expect(Token::Paren('>'))?;
        Ok(format)
    }

    pub(super) fn open_arguments(&mut self) -> Result<(), Error<'a>> {
        self.expect(Token::Paren('('))
    }

    pub(super) fn close_arguments(&mut self) -> Result<(), Error<'a>> {
        let _ = self.skip(Token::Separator(','));
        self.expect(Token::Paren(')'))
    }

    pub(super) fn next_argument(&mut self) -> Result<bool, Error<'a>> {
        let paren = Token::Paren(')');
        if self.skip(Token::Separator(',')) {
            Ok(!self.skip(paren))
        } else {
            self.expect(paren).map(|()| false)
        }
    }
}

#[cfg(test)]
fn sub_test(source: &str, expected_tokens: &[Token]) {
    let mut lex = Lexer::new(source);
    for &token in expected_tokens {
        assert_eq!(lex.next().0, token);
    }
    assert_eq!(lex.next().0, Token::End);
}

#[test]
fn test_tokens() {
    sub_test("id123_OK", &[Token::Word("id123_OK")]);
    sub_test(
        "92No",
        &[
            Token::Number {
                value: "92",
                ty: 'i',
                width: "",
            },
            Token::Word("No"),
        ],
    );
    sub_test(
        "2u3o",
        &[
            Token::Number {
                value: "2",
                ty: 'u',
                width: "3",
            },
            Token::Word("o"),
        ],
    );
    sub_test(
        "2.4f44po",
        &[
            Token::Number {
                value: "2.4",
                ty: 'f',
                width: "44",
            },
            Token::Word("po"),
        ],
    );
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
            Token::Number {
                value: "0",
                ty: 'i',
                width: "",
            },
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
