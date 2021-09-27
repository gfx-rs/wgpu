use super::{conv, Error, ExpectedToken, NumberType, Span, Token, TokenSpan};

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

/// Tries to skip a given prefix in the input string.
/// Returns whether the prefix was present and could therefore be skipped,
/// the remaining str and the number of *bytes* skipped.
pub fn try_skip_prefix<'a, 'b>(input: &'a str, prefix: &'b str) -> (bool, &'a str, usize) {
    if input.starts_with(prefix) {
        (true, &input[prefix.len()..], prefix.len())
    } else {
        (false, input, 0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NLDigitState {
    Nothing,
    LeadingZero,
    DigitBeforeDot,
    OnlyDot,
    DigitsThenDot,
    DigitAfterDot,
    Exponent,
    SignAfterExponent,
    DigitAfterExponent,
}

struct NumberLexerState {
    _minus: bool,
    hex: bool,
    leading_zeros: usize,
    digit_state: NLDigitState,
    uint_suffix: bool,
}

impl NumberLexerState {
    // TODO: add proper error reporting, possibly through try_into_token function returning Result

    pub fn _is_valid_number(&self) -> bool {
        match *self {
            Self {
                _minus: false, // No negative zero for integers.
                hex,
                leading_zeros,
                digit_state: NLDigitState::LeadingZero,
                ..
            } => hex || leading_zeros == 1, // No leading zeros allowed in non-hex integers, "0" is always allowed.
            Self {
                _minus: minus,
                hex,
                leading_zeros,
                digit_state: NLDigitState::DigitBeforeDot,
                uint_suffix,
            } => {
                (hex || leading_zeros == 0) // No leading zeros allowed in non-hex integers.
                                              // In this state the number has non-zero digits,
                                              // i.e. it is not just "0".
                    && (minus ^ uint_suffix) // Either a negative number, or and unsigned integer, not both.
            }
            _ => self.is_float(),
        }
    }

    pub fn is_float(&self) -> bool {
        !self.uint_suffix
            && (self.digit_state == NLDigitState::DigitsThenDot
                || self.digit_state == NLDigitState::DigitAfterDot
                || self.digit_state == NLDigitState::DigitAfterExponent)
    }
}

fn consume_number(input: &str) -> (Token, &str) {
    let (minus, working_substr, minus_offset) = try_skip_prefix(input, "-");

    let (hex, working_substr, hex_offset) = try_skip_prefix(working_substr, "0x");

    let mut state = NumberLexerState {
        _minus: minus,
        hex,
        leading_zeros: 0,
        digit_state: NLDigitState::Nothing,
        uint_suffix: false,
    };

    let mut what = |c| {
        match state {
            NumberLexerState {
                hex,
                digit_state: NLDigitState::Nothing,
                uint_suffix: false,
                ..
            } => match c {
                '0' => {
                    state.digit_state = NLDigitState::LeadingZero;
                    state.leading_zeros += 1;
                }
                '1'..='9' => {
                    state.digit_state = NLDigitState::DigitBeforeDot;
                }
                'a'..='f' | 'A'..='F' if hex => {
                    state.digit_state = NLDigitState::DigitBeforeDot;
                }
                '.' => {
                    state.digit_state = NLDigitState::OnlyDot;
                }
                _ => return false,
            },

            NumberLexerState {
                hex,
                digit_state: NLDigitState::LeadingZero,
                uint_suffix: false,
                ..
            } => match c {
                '0' => {
                    // We stay in NLDigitState::LeadingZero.
                    state.leading_zeros += 1;
                }
                '1'..='9' => {
                    state.digit_state = NLDigitState::DigitBeforeDot;
                }
                'a'..='f' | 'A'..='F' if hex => {
                    state.digit_state = NLDigitState::DigitBeforeDot;
                }
                '.' => {
                    state.digit_state = NLDigitState::DigitsThenDot;
                }
                'e' | 'E' if !hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                'p' | 'P' if hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                'u' => {
                    // We stay in NLDigitState::LeadingZero.
                    state.uint_suffix = true;
                }
                _ => return false,
            },

            NumberLexerState {
                hex,
                digit_state: NLDigitState::DigitBeforeDot,
                uint_suffix: false,
                ..
            } => match c {
                '0'..='9' => {
                    // We stay in NLDigitState::DigitBeforeDot.
                }
                'a'..='f' | 'A'..='F' if hex => {
                    // We stay in NLDigitState::DigitBeforeDot.
                }
                '.' => {
                    state.digit_state = NLDigitState::DigitsThenDot;
                }
                'e' | 'E' if !hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                'p' | 'P' if hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                'u' => {
                    // We stay in NLDigitState::DigitBeforeDot.
                    state.uint_suffix = true;
                }
                _ => return false,
            },

            NumberLexerState {
                hex,
                digit_state: NLDigitState::OnlyDot,
                uint_suffix: false,
                ..
            } => match c {
                '0'..='9' => {
                    state.digit_state = NLDigitState::DigitAfterDot;
                }
                'a'..='f' | 'A'..='F' if hex => {
                    state.digit_state = NLDigitState::DigitAfterDot;
                }
                _ => return false,
            },

            NumberLexerState {
                hex,
                digit_state: NLDigitState::DigitsThenDot,
                uint_suffix: false,
                ..
            }
            | NumberLexerState {
                hex,
                digit_state: NLDigitState::DigitAfterDot,
                uint_suffix: false,
                ..
            } => match c {
                '0'..='9' => {
                    state.digit_state = NLDigitState::DigitAfterDot;
                }
                'a'..='f' | 'A'..='F' if hex => {
                    state.digit_state = NLDigitState::DigitAfterDot;
                }
                'e' | 'E' if !hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                'p' | 'P' if hex => {
                    state.digit_state = NLDigitState::Exponent;
                }
                _ => return false,
            },

            NumberLexerState {
                digit_state: NLDigitState::Exponent,
                uint_suffix: false,
                ..
            } => match c {
                '0'..='9' => {
                    state.digit_state = NLDigitState::DigitAfterExponent;
                }
                '-' | '+' => {
                    state.digit_state = NLDigitState::SignAfterExponent;
                }
                _ => return false,
            },

            NumberLexerState {
                digit_state: NLDigitState::SignAfterExponent,
                uint_suffix: false,
                ..
            }
            | NumberLexerState {
                digit_state: NLDigitState::DigitAfterExponent,
                uint_suffix: false,
                ..
            } => match c {
                '0'..='9' => {
                    state.digit_state = NLDigitState::DigitAfterExponent;
                }
                _ => return false,
            },

            NumberLexerState {
                uint_suffix: true, ..
            } => return false, // Scanning is done once we've reached a type suffix.
        }

        // No match branch has rejected this yet, so we are still in a number literal
        true
    };

    let pos = working_substr
        .find(|c| !what(c))
        .unwrap_or_else(|| working_substr.len());
    let (value, rest) = input.split_at(pos + minus_offset + hex_offset);

    // NOTE: This code can use string slicing,
    //       because number literals are exclusively ASCII.
    //       This means all relevant characters fit into one byte
    //       and using string slicing (which slices UTF-8 bytes) works for us.

    // TODO: A syntax error can already be recognized here, possibly report it at this stage.

    // Return possibly knowably incorrect (given !state.is_valid_number()) token for now.
    (
        Token::Number {
            value: if state.uint_suffix {
                &value[..value.len() - 1]
            } else {
                value
            },
            ty: if state.uint_suffix {
                NumberType::Uint
            } else if state.is_float() {
                NumberType::Float
            } else {
                NumberType::Sint
            },
            width: None,
        },
        rest,
    )
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
        'a'..='z' | 'A'..='Z' => {
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

    /// Calls the function with a lexer and returns the result of the function as well as the span for everything the function parsed
    ///
    /// # Examples
    /// ```ignore
    /// let lexer = Lexer::new("5");
    /// let (value, span) = lexer.capture_span(Lexer::next_uint_literal);
    /// assert_eq!(value, 5);
    /// ```
    #[inline]
    pub fn capture_span<T, E>(
        &mut self,
        inner: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<(T, Span), E> {
        let start = self.current_byte_offset();
        let res = inner(self)?;
        let end = self.current_byte_offset();
        Ok((res, start..end))
    }

    fn peek_token_and_rest(&mut self) -> (TokenSpan<'a>, &'a str) {
        let mut cloned = self.clone();
        let token = cloned.next();
        let rest = cloned.input;
        (token, rest)
    }

    pub(super) fn current_byte_offset(&self) -> usize {
        self.source.len() - self.input.len()
    }

    pub(super) fn span_from(&self, offset: usize) -> Span {
        offset..self.current_byte_offset()
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

    pub(super) fn expect_span(
        &mut self,
        expected: Token<'a>,
    ) -> Result<std::ops::Range<usize>, Error<'a>> {
        let next = self.next();
        if next.0 == expected {
            Ok(next.1)
        } else {
            Err(Error::Unexpected(next, ExpectedToken::Token(expected)))
        }
    }

    pub(super) fn expect(&mut self, expected: Token<'a>) -> Result<(), Error<'a>> {
        self.expect_span(expected)?;
        Ok(())
    }

    pub(super) fn expect_generic_paren(&mut self, expected: char) -> Result<(), Error<'a>> {
        let next = self.next_generic();
        if next.0 == Token::Paren(expected) {
            Ok(())
        } else {
            Err(Error::Unexpected(
                next,
                ExpectedToken::Token(Token::Paren(expected)),
            ))
        }
    }

    /// If the next token matches it is skipped and true is returned
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
            other => Err(Error::Unexpected(other, ExpectedToken::Identifier)),
        }
    }

    pub(super) fn next_ident(&mut self) -> Result<&'a str, Error<'a>> {
        match self.next() {
            (Token::Word(word), _) => Ok(word),
            other => Err(Error::Unexpected(other, ExpectedToken::Identifier)),
        }
    }

    /// Parses a generic scalar type, for example `<f32>`.
    pub(super) fn next_scalar_generic(
        &mut self,
    ) -> Result<(crate::ScalarKind, crate::Bytes), Error<'a>> {
        self.expect_generic_paren('<')?;
        let pair = match self.next() {
            (Token::Word(word), span) => {
                conv::get_scalar_type(word).ok_or(Error::UnknownScalarType(span))
            }
            (_, span) => Err(Error::UnknownScalarType(span)),
        }?;
        self.expect_generic_paren('>')?;
        Ok(pair)
    }

    /// Parses a generic scalar type, for example `<f32>`.
    ///
    /// Returns the span covering the inner type, excluding the brackets.
    pub(super) fn next_scalar_generic_with_span(
        &mut self,
    ) -> Result<(crate::ScalarKind, crate::Bytes, Span), Error<'a>> {
        self.expect_generic_paren('<')?;
        let pair = match self.next() {
            (Token::Word(word), span) => conv::get_scalar_type(word)
                .map(|(a, b)| (a, b, span.clone()))
                .ok_or(Error::UnknownScalarType(span)),
            (_, span) => Err(Error::UnknownScalarType(span)),
        }?;
        self.expect_generic_paren('>')?;
        Ok(pair)
    }

    pub(super) fn next_storage_access(&mut self) -> Result<crate::StorageAccess, Error<'a>> {
        let (ident, span) = self.next_ident_with_span()?;
        match ident {
            "read" => Ok(crate::StorageAccess::LOAD),
            "write" => Ok(crate::StorageAccess::STORE),
            "read_write" => Ok(crate::StorageAccess::LOAD | crate::StorageAccess::STORE),
            _ => Err(Error::UnknownAccess(span)),
        }
    }

    pub(super) fn next_format_generic(
        &mut self,
    ) -> Result<(crate::StorageFormat, crate::StorageAccess), Error<'a>> {
        self.expect(Token::Paren('<'))?;
        let (ident, ident_span) = self.next_ident_with_span()?;
        let format = conv::map_storage_format(ident, ident_span)?;
        self.expect(Token::Separator(','))?;
        let access = self.next_storage_access()?;
        self.expect(Token::Paren('>'))?;
        Ok((format, access))
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
                ty: NumberType::Sint,
                width: None,
            },
            Token::Word("No"),
        ],
    );
    sub_test(
        "2u3o",
        &[
            Token::Number {
                value: "2",
                ty: NumberType::Uint,
                width: None,
            },
            Token::Number {
                value: "3",
                ty: NumberType::Sint,
                width: None,
            },
            Token::Word("o"),
        ],
    );
    sub_test(
        "2.4f44po",
        &[
            Token::Number {
                value: "2.4",
                ty: NumberType::Float,
                width: None,
            },
            Token::Word("f44po"),
        ],
    );
    sub_test(
        "æNoø",
        &[Token::Unknown('æ'), Token::Word("No"), Token::Unknown('ø')],
    );
    sub_test("No¾", &[Token::Word("No"), Token::Unknown('¾')]);
    sub_test("No好", &[Token::Word("No"), Token::Unknown('好')]);
    sub_test("_No", &[Token::Unknown('_'), Token::Word("No")]);
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
                ty: NumberType::Sint,
                width: None,
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
    );
    sub_test(
        "var<storage,read_write> buffer: array<u32>;",
        &[
            Token::Word("var"),
            Token::Paren('<'),
            Token::Word("storage"),
            Token::Separator(','),
            Token::Word("read_write"),
            Token::Paren('>'),
            Token::Word("buffer"),
            Token::Separator(':'),
            Token::Word("array"),
            Token::Paren('<'),
            Token::Word("u32"),
            Token::Paren('>'),
            Token::Separator(';'),
        ],
    );
}
