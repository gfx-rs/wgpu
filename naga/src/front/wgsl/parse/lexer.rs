use super::{number::consume_number, Error, ExpectedToken};
use crate::front::wgsl::error::NumberError;
use crate::front::wgsl::parse::{conv, Number};
use crate::front::wgsl::Scalar;
use crate::Span;
type TokenSpan<'a> = (Token<'a>, Span);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    Paren(char),
    Attribute,
    Number(Result<Number, NumberError>),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    AssignmentOperation(char),
    IncrementOperation,
    DecrementOperation,
    Arrow,
    Unknown(char),
    Trivia,
    Comment(&'a str),
    End,
}

fn consume_any(input: &str, what: impl Fn(char) -> bool) -> (&str, &str) {
    let pos = input.find(|c| !what(c)).unwrap_or(input.len());
    input.split_at(pos)
}

/// Return the token at the start of `input`.
///
/// If `generic` is `false`, then the bit shift operators `>>` or `<<`
/// are valid lookahead tokens for the current parser state (see [§3.1
/// Parsing] in the WGSL specification). In other words:
///
/// -   If `generic` is `true`, then we are expecting an angle bracket
///     around a generic type parameter, like the `<` and `>` in
///     `vec3<f32>`, so interpret `<` and `>` as `Token::Paren` tokens,
///     even if they're part of `<<` or `>>` sequences.
///
/// -   Otherwise, interpret `<<` and `>>` as shift operators:
///     `Token::LogicalOperation` tokens.
///
/// [§3.1 Parsing]: https://gpuweb.github.io/gpuweb/wgsl/#parsing
fn consume_token(input: &str, generic: bool) -> (Token<'_>, &str) {
    let mut chars = input.chars();
    let full_chars = input.chars();
    let cur = match chars.next() {
        Some(c) => c,
        None => return (Token::End, ""),
    };
    match cur {
        ':' | ';' | ',' => (Token::Separator(cur), chars.as_str()),
        '.' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('0'..='9') => consume_number(input),
                _ => (Token::Separator(cur), og_chars),
            }
        }
        '@' => (Token::Attribute, chars.as_str()),
        '(' | ')' | '{' | '}' | '[' | ']' => (Token::Paren(cur), chars.as_str()),
        '<' | '>' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('=') if !generic => (Token::LogicalOperation(cur), chars.as_str()),
                Some(c) if c == cur && !generic => {
                    let og_chars = chars.as_str();
                    match chars.next() {
                        Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                        _ => (Token::ShiftOperation(cur), og_chars),
                    }
                }
                _ => (Token::Paren(cur), og_chars),
            }
        }
        '0'..='9' => consume_number(input),
        '/' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('/') => {
                    if let Some(end_position) = input
                        .char_indices()
                        .find(|char_indices| is_comment_end(char_indices.1))
                    {
                        let end_position = end_position.0 + 1;
                        return (
                            Token::Comment(&input[..end_position]),
                            &input[end_position..],
                        );
                    }
                    (Token::Comment(input), &input[input.len() - 1..])
                }
                Some('*') => {
                    let mut depth = 1;
                    let mut prev = None;
                    let mut nb_characters = 2;
                    for c in &mut chars {
                        match (prev, c) {
                            (Some('*'), '/') => {
                                prev = None;
                                depth -= 1;
                                nb_characters += 1;
                                if depth == 0 {
                                    let doc = &full_chars.as_str()[..nb_characters];
                                    return (Token::Comment(doc), chars.as_str());
                                }
                            }
                            (Some('/'), '*') => {
                                prev = None;
                                depth += 1;
                                nb_characters += 1;
                            }
                            _ => {
                                prev = Some(c);
                                nb_characters += 1;
                            }
                        }
                    }

                    (Token::End, "")
                }
                Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '-' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('>') => (Token::Arrow, chars.as_str()),
                Some('-') => (Token::DecrementOperation, chars.as_str()),
                Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '+' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('+') => (Token::IncrementOperation, chars.as_str()),
                Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '*' | '%' | '^' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '~' => (Token::Operation(cur), chars.as_str()),
        '=' | '!' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some('=') => (Token::LogicalOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        '&' | '|' => {
            let og_chars = chars.as_str();
            match chars.next() {
                Some(c) if c == cur => (Token::LogicalOperation(cur), chars.as_str()),
                Some('=') => (Token::AssignmentOperation(cur), chars.as_str()),
                _ => (Token::Operation(cur), og_chars),
            }
        }
        _ if is_blankspace(cur) => {
            let (_, rest) = consume_any(input, is_blankspace);
            (Token::Trivia, rest)
        }
        _ if is_word_start(cur) => {
            let (word, rest) = consume_any(input, is_word_part);
            (Token::Word(word), rest)
        }
        _ => (Token::Unknown(cur), chars.as_str()),
    }
}

/// Returns whether or not a char is a comment end
/// (Unicode Pattern_White_Space excluding U+0020, U+0009, U+200E and U+200F)
/// https://www.w3.org/TR/WGSL/#line-break
const fn is_comment_end(c: char) -> bool {
    match c {
        '\u{000a}'..='\u{000d}' | '\u{0085}' | '\u{2028}' | '\u{2029}' => true,
        _ => false,
    }
}

/// Returns whether or not a char is a blankspace (Unicode Pattern_White_Space)
const fn is_blankspace(c: char) -> bool {
    match c {
        '\u{0020}'
        | '\u{0009}'..='\u{000d}'
        | '\u{0085}'
        | '\u{200e}'
        | '\u{200f}'
        | '\u{2028}'
        | '\u{2029}' => true,
        _ => false,
    }
}

/// Returns whether or not a char is a word start (Unicode XID_Start + '_')
fn is_word_start(c: char) -> bool {
    c == '_' || unicode_xid::UnicodeXID::is_xid_start(c)
}

/// Returns whether or not a char is a word part (Unicode XID_Continue)
fn is_word_part(c: char) -> bool {
    unicode_xid::UnicodeXID::is_xid_continue(c)
}

#[derive(Clone)]
pub(in crate::front::wgsl) struct Lexer<'a> {
    input: &'a str,
    pub(in crate::front::wgsl) source: &'a str,
    // The byte offset of the end of the last non-trivia token.
    last_end_offset: usize,
}

impl<'a> Lexer<'a> {
    pub(in crate::front::wgsl) const fn new(input: &'a str) -> Self {
        Lexer {
            input,
            source: input,
            last_end_offset: 0,
        }
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
        Ok((res, Span::from(start..end)))
    }

    pub(in crate::front::wgsl) fn start_byte_offset(&mut self) -> usize {
        loop {
            // Eat all trivia because `next` doesn't eat trailing trivia.
            let (token, rest) = consume_token(self.input, false);
            if let Token::Trivia | Token::Comment(_) = token {
                self.input = rest;
            } else {
                return self.current_byte_offset();
            }
        }
    }

    fn peek_token_and_rest(&mut self) -> (TokenSpan<'a>, &'a str) {
        let mut cloned = self.clone();
        let token = cloned.next();
        let rest = cloned.input;
        (token, rest)
    }

    pub(in crate::front::wgsl) fn start_byte_offset_and_aggregate_comment(
        &'a mut self,
        comments: &mut Vec<Span>,
    ) -> usize {
        loop {
            let start = self.current_byte_offset();
            // Eat all trivia because `next` doesn't eat trailing trivia.
            let (token, rest) = consume_token(self.input, false);
            if let Token::Comment(_) = token {
                self.input = rest;
                let next = self.current_byte_offset();
                comments.push(Span::new(start as u32, next as u32));
            } else if let Token::Trivia = token {
                self.input = rest;
            } else {
                return self.current_byte_offset();
            }
        }
    }

    pub const fn current_byte_offset(&self) -> usize {
        self.source.len() - self.input.len()
    }

    pub(in crate::front::wgsl) fn span_from(&self, offset: usize) -> Span {
        Span::from(offset..self.last_end_offset)
    }

    /// Return the next non-whitespace token from `self`.
    ///
    /// Assume we are a parse state where bit shift operators may
    /// occur, but not angle brackets.
    #[must_use]
    pub(in crate::front::wgsl) fn next(&mut self) -> TokenSpan<'a> {
        self.next_impl(false)
    }

    /// Return the next non-whitespace token from `self`.
    ///
    /// Assume we are in a parse state where angle brackets may occur,
    /// but not bit shift operators.
    #[must_use]
    pub(in crate::front::wgsl) fn next_generic(&mut self) -> TokenSpan<'a> {
        self.next_impl(true)
    }

    /// Return the next non-whitespace token from `self`, with a span.
    ///
    /// See [`consume_token`] for the meaning of `generic`.
    fn next_impl(&mut self, generic: bool) -> TokenSpan<'a> {
        self.next_until(
            |token| !matches!(token, Token::Trivia | Token::Comment(_)),
            generic,
        )
    }

    /// Return the next token from `self` for which `stop_at` returns true.
    ///
    /// See [`consume_token`] for the meaning of `generic`.
    pub fn next_until(&mut self, stop_at: fn(Token) -> bool, generic: bool) -> TokenSpan<'a> {
        let mut start_byte_offset = self.current_byte_offset();
        loop {
            let (token, rest) = consume_token(self.input, generic);
            self.input = rest;
            if stop_at(token) {
                self.last_end_offset = self.current_byte_offset();
                return (token, self.span_from(start_byte_offset));
            }
            start_byte_offset = self.current_byte_offset();
        }
    }

    #[must_use]
    pub(in crate::front::wgsl) fn peek(&mut self) -> TokenSpan<'a> {
        let (token, _) = self.peek_token_and_rest();
        token
    }

    pub(in crate::front::wgsl) fn expect_span(
        &mut self,
        expected: Token<'a>,
    ) -> Result<Span, Error<'a>> {
        let next = self.next();
        if next.0 == expected {
            Ok(next.1)
        } else {
            Err(Error::Unexpected(next.1, ExpectedToken::Token(expected)))
        }
    }

    pub(in crate::front::wgsl) fn expect(&mut self, expected: Token<'a>) -> Result<(), Error<'a>> {
        self.expect_span(expected)?;
        Ok(())
    }

    pub(in crate::front::wgsl) fn expect_generic_paren(
        &mut self,
        expected: char,
    ) -> Result<(), Error<'a>> {
        let next = self.next_generic();
        if next.0 == Token::Paren(expected) {
            Ok(())
        } else {
            Err(Error::Unexpected(
                next.1,
                ExpectedToken::Token(Token::Paren(expected)),
            ))
        }
    }

    /// If the next token matches it is skipped and true is returned
    pub(in crate::front::wgsl) fn skip(&mut self, what: Token<'_>) -> bool {
        let (peeked_token, rest) = self.peek_token_and_rest();
        if peeked_token.0 == what {
            self.input = rest;
            true
        } else {
            false
        }
    }

    pub(in crate::front::wgsl) fn next_ident_with_span(
        &mut self,
    ) -> Result<(&'a str, Span), Error<'a>> {
        match self.next() {
            (Token::Word("_"), span) => Err(Error::InvalidIdentifierUnderscore(span)),
            (Token::Word(word), span) if word.starts_with("__") => {
                Err(Error::ReservedIdentifierPrefix(span))
            }
            (Token::Word(word), span) => Ok((word, span)),
            other => Err(Error::Unexpected(other.1, ExpectedToken::Identifier)),
        }
    }

    pub(in crate::front::wgsl) fn next_ident(
        &mut self,
    ) -> Result<super::ast::Ident<'a>, Error<'a>> {
        let ident = self
            .next_ident_with_span()
            .map(|(name, span)| super::ast::Ident { name, span })?;

        if crate::keywords::wgsl::RESERVED.contains(&ident.name) {
            return Err(Error::ReservedKeyword(ident.span));
        }

        Ok(ident)
    }

    /// Parses a generic scalar type, for example `<f32>`.
    pub(in crate::front::wgsl) fn next_scalar_generic(&mut self) -> Result<Scalar, Error<'a>> {
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
    pub(in crate::front::wgsl) fn next_scalar_generic_with_span(
        &mut self,
    ) -> Result<(Scalar, Span), Error<'a>> {
        self.expect_generic_paren('<')?;
        let pair = match self.next() {
            (Token::Word(word), span) => conv::get_scalar_type(word)
                .map(|scalar| (scalar, span))
                .ok_or(Error::UnknownScalarType(span)),
            (_, span) => Err(Error::UnknownScalarType(span)),
        }?;
        self.expect_generic_paren('>')?;
        Ok(pair)
    }

    pub(in crate::front::wgsl) fn next_storage_access(
        &mut self,
    ) -> Result<crate::StorageAccess, Error<'a>> {
        let (ident, span) = self.next_ident_with_span()?;
        match ident {
            "read" => Ok(crate::StorageAccess::LOAD),
            "write" => Ok(crate::StorageAccess::STORE),
            "read_write" => Ok(crate::StorageAccess::LOAD | crate::StorageAccess::STORE),
            _ => Err(Error::UnknownAccess(span)),
        }
    }

    pub(in crate::front::wgsl) fn next_format_generic(
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

    pub(in crate::front::wgsl) fn open_arguments(&mut self) -> Result<(), Error<'a>> {
        self.expect(Token::Paren('('))
    }

    pub(in crate::front::wgsl) fn close_arguments(&mut self) -> Result<(), Error<'a>> {
        let _ = self.skip(Token::Separator(','));
        self.expect(Token::Paren(')'))
    }

    pub(in crate::front::wgsl) fn next_argument(&mut self) -> Result<bool, Error<'a>> {
        let paren = Token::Paren(')');
        if self.skip(Token::Separator(',')) {
            Ok(!self.skip(paren))
        } else {
            self.expect(paren).map(|()| false)
        }
    }
}

#[cfg(test)]
#[track_caller]
fn sub_test(source: &str, expected_tokens: &[Token]) {
    let mut lex = Lexer::new(source);
    for &token in expected_tokens {
        assert_eq!(
            lex.next_until(|token| !matches!(token, Token::Trivia), false)
                .0,
            token
        );
    }
    assert_eq!(lex.next().0, Token::End);
}

#[test]
fn test_numbers() {
    // WGSL spec examples //

    // decimal integer
    sub_test(
        "0x123 0X123u 1u 123 0 0i 0x3f",
        &[
            Token::Number(Ok(Number::AbstractInt(291))),
            Token::Number(Ok(Number::U32(291))),
            Token::Number(Ok(Number::U32(1))),
            Token::Number(Ok(Number::AbstractInt(123))),
            Token::Number(Ok(Number::AbstractInt(0))),
            Token::Number(Ok(Number::I32(0))),
            Token::Number(Ok(Number::AbstractInt(63))),
        ],
    );
    // decimal floating point
    sub_test(
        "0.e+4f 01. .01 12.34 .0f 0h 1e-3 0xa.fp+2 0x1P+4f 0X.3 0x3p+2h 0X1.fp-4 0x3.2p+2h",
        &[
            Token::Number(Ok(Number::F32(0.))),
            Token::Number(Ok(Number::AbstractFloat(1.))),
            Token::Number(Ok(Number::AbstractFloat(0.01))),
            Token::Number(Ok(Number::AbstractFloat(12.34))),
            Token::Number(Ok(Number::F32(0.))),
            Token::Number(Err(NumberError::UnimplementedF16)),
            Token::Number(Ok(Number::AbstractFloat(0.001))),
            Token::Number(Ok(Number::AbstractFloat(43.75))),
            Token::Number(Ok(Number::F32(16.))),
            Token::Number(Ok(Number::AbstractFloat(0.1875))),
            Token::Number(Err(NumberError::UnimplementedF16)),
            Token::Number(Ok(Number::AbstractFloat(0.12109375))),
            Token::Number(Err(NumberError::UnimplementedF16)),
        ],
    );

    // MIN / MAX //

    // min / max decimal integer
    sub_test(
        "0i 2147483647i 2147483648i",
        &[
            Token::Number(Ok(Number::I32(0))),
            Token::Number(Ok(Number::I32(i32::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );
    // min / max decimal unsigned integer
    sub_test(
        "0u 4294967295u 4294967296u",
        &[
            Token::Number(Ok(Number::U32(u32::MIN))),
            Token::Number(Ok(Number::U32(u32::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );

    // min / max hexadecimal signed integer
    sub_test(
        "0x0i 0x7FFFFFFFi 0x80000000i",
        &[
            Token::Number(Ok(Number::I32(0))),
            Token::Number(Ok(Number::I32(i32::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );
    // min / max hexadecimal unsigned integer
    sub_test(
        "0x0u 0xFFFFFFFFu 0x100000000u",
        &[
            Token::Number(Ok(Number::U32(u32::MIN))),
            Token::Number(Ok(Number::U32(u32::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );

    // min/max decimal abstract int
    sub_test(
        "0 9223372036854775807 9223372036854775808",
        &[
            Token::Number(Ok(Number::AbstractInt(0))),
            Token::Number(Ok(Number::AbstractInt(i64::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );

    // min/max hexadecimal abstract int
    sub_test(
        "0 0x7fffffffffffffff 0x8000000000000000",
        &[
            Token::Number(Ok(Number::AbstractInt(0))),
            Token::Number(Ok(Number::AbstractInt(i64::MAX))),
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );

    /// ≈ 2^-126 * 2^−23 (= 2^−149)
    const SMALLEST_POSITIVE_SUBNORMAL_F32: f32 = 1e-45;
    /// ≈ 2^-126 * (1 − 2^−23)
    const LARGEST_SUBNORMAL_F32: f32 = 1.1754942e-38;
    /// ≈ 2^-126
    const SMALLEST_POSITIVE_NORMAL_F32: f32 = f32::MIN_POSITIVE;
    /// ≈ 1 − 2^−24
    const LARGEST_F32_LESS_THAN_ONE: f32 = 0.99999994;
    /// ≈ 1 + 2^−23
    const SMALLEST_F32_LARGER_THAN_ONE: f32 = 1.0000001;
    /// ≈ 2^127 * (2 − 2^−23)
    const LARGEST_NORMAL_F32: f32 = f32::MAX;

    // decimal floating point
    sub_test(
        "1e-45f 1.1754942e-38f 1.17549435e-38f 0.99999994f 1.0000001f 3.40282347e+38f",
        &[
            Token::Number(Ok(Number::F32(SMALLEST_POSITIVE_SUBNORMAL_F32))),
            Token::Number(Ok(Number::F32(LARGEST_SUBNORMAL_F32))),
            Token::Number(Ok(Number::F32(SMALLEST_POSITIVE_NORMAL_F32))),
            Token::Number(Ok(Number::F32(LARGEST_F32_LESS_THAN_ONE))),
            Token::Number(Ok(Number::F32(SMALLEST_F32_LARGER_THAN_ONE))),
            Token::Number(Ok(Number::F32(LARGEST_NORMAL_F32))),
        ],
    );
    sub_test(
        "3.40282367e+38f",
        &[
            Token::Number(Err(NumberError::NotRepresentable)), // ≈ 2^128
        ],
    );

    // hexadecimal floating point
    sub_test(
        "0x1p-149f 0x7FFFFFp-149f 0x1p-126f 0xFFFFFFp-24f 0x800001p-23f 0xFFFFFFp+104f",
        &[
            Token::Number(Ok(Number::F32(SMALLEST_POSITIVE_SUBNORMAL_F32))),
            Token::Number(Ok(Number::F32(LARGEST_SUBNORMAL_F32))),
            Token::Number(Ok(Number::F32(SMALLEST_POSITIVE_NORMAL_F32))),
            Token::Number(Ok(Number::F32(LARGEST_F32_LESS_THAN_ONE))),
            Token::Number(Ok(Number::F32(SMALLEST_F32_LARGER_THAN_ONE))),
            Token::Number(Ok(Number::F32(LARGEST_NORMAL_F32))),
        ],
    );
    sub_test(
        "0x1p128f 0x1.000001p0f",
        &[
            Token::Number(Err(NumberError::NotRepresentable)), // = 2^128
            Token::Number(Err(NumberError::NotRepresentable)),
        ],
    );
}

#[test]
fn double_floats() {
    sub_test(
        "0x1.2p4lf 0x1p8lf 0.0625lf 625e-4lf 10lf 10l",
        &[
            Token::Number(Ok(Number::F64(18.0))),
            Token::Number(Ok(Number::F64(256.0))),
            Token::Number(Ok(Number::F64(0.0625))),
            Token::Number(Ok(Number::F64(0.0625))),
            Token::Number(Ok(Number::F64(10.0))),
            Token::Number(Ok(Number::AbstractInt(10))),
            Token::Word("l"),
        ],
    )
}

#[test]
fn test_tokens() {
    sub_test("id123_OK", &[Token::Word("id123_OK")]);
    sub_test(
        "92No",
        &[
            Token::Number(Ok(Number::AbstractInt(92))),
            Token::Word("No"),
        ],
    );
    sub_test(
        "2u3o",
        &[
            Token::Number(Ok(Number::U32(2))),
            Token::Number(Ok(Number::AbstractInt(3))),
            Token::Word("o"),
        ],
    );
    sub_test(
        "2.4f44po",
        &[
            Token::Number(Ok(Number::F32(2.4))),
            Token::Number(Ok(Number::AbstractInt(44))),
            Token::Word("po"),
        ],
    );
    sub_test(
        "Δέλτα réflexion Кызыл 𐰓𐰏𐰇 朝焼け سلام 검정 שָׁלוֹם गुलाबी փիրուզ",
        &[
            Token::Word("Δέλτα"),
            Token::Word("réflexion"),
            Token::Word("Кызыл"),
            Token::Word("𐰓𐰏𐰇"),
            Token::Word("朝焼け"),
            Token::Word("سلام"),
            Token::Word("검정"),
            Token::Word("שָׁלוֹם"),
            Token::Word("गुलाबी"),
            Token::Word("փիրուզ"),
        ],
    );
    sub_test("æNoø", &[Token::Word("æNoø")]);
    sub_test("No¾", &[Token::Word("No"), Token::Unknown('¾')]);
    sub_test("No好", &[Token::Word("No好")]);
    sub_test("_No", &[Token::Word("_No")]);
    sub_test(
        "*/*/***/*//=/*****//",
        &[
            Token::Operation('*'),
            Token::AssignmentOperation('/'),
            Token::Operation('/'),
        ],
    );

    // Type suffixes are only allowed on hex float literals
    // if you provided an exponent.
    sub_test(
        "0x1.2f 0x1.2f 0x1.2h 0x1.2H 0x1.2lf",
        &[
            // The 'f' suffixes are taken as a hex digit:
            // the fractional part is 0x2f / 256.
            Token::Number(Ok(Number::AbstractFloat(1.0 + 0x2f as f64 / 256.0))),
            Token::Number(Ok(Number::AbstractFloat(1.0 + 0x2f as f64 / 256.0))),
            Token::Number(Ok(Number::AbstractFloat(1.125))),
            Token::Word("h"),
            Token::Number(Ok(Number::AbstractFloat(1.125))),
            Token::Word("H"),
            Token::Number(Ok(Number::AbstractFloat(1.125))),
            Token::Word("lf"),
        ],
    )
}

#[test]
fn test_variable_decl() {
    sub_test(
        "@group(0 ) var< uniform> texture:   texture_multisampled_2d <f32 >;",
        &[
            Token::Attribute,
            Token::Word("group"),
            Token::Paren('('),
            Token::Number(Ok(Number::AbstractInt(0))),
            Token::Paren(')'),
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

#[test]
fn test_comments() {
    sub_test("// Single comment", &[Token::Comment("// Single comment")]);
    sub_test(
        "/* multi
    line
    comment */",
        &[Token::Comment(
            "/* multi
    line
    comment */",
        )],
    );
    sub_test(
        "/* multi
    line
    comment */
    // and another",
        &[
            Token::Comment(
                "/* multi
    line
    comment */",
            ),
            Token::Comment("// and another"),
        ],
    );
}

#[test]
fn test_comment_nested() {
    sub_test(
        "/*
    a comment with nested one /*
        nested comment
    */
    */
    const a : i32 = 2;",
        &[
            Token::Comment(
                "/*
    a comment with nested one /*
        nested comment
    */
    */",
            ),
            Token::Word("const"),
            Token::Word("a"),
            Token::Separator(':'),
            Token::Word("i32"),
            Token::Operation('='),
            Token::Number(Ok(Number::AbstractInt(2))),
            Token::Separator(';'),
        ],
    );
}

#[test]
fn test_comment_long_character() {
    sub_test(
        "// π/2
        //     D(𝐡) = ───────────────────────────────────────────────────
//            παₜα_b((𝐡 ⋅ 𝐭)² / αₜ²) + (𝐡 ⋅ 𝐛)² / α_b² +`
    const a : i32 = 2;",
        &[
            Token::Comment("// π/2"),
            Token::Comment("//     D(𝐡) = ───────────────────────────────────────────────────"),
            Token::Comment("//            παₜα_b((𝐡 ⋅ 𝐭)² / αₜ²) + (𝐡 ⋅ 𝐛)² / α_b² +`"),
            Token::Word("const"),
            Token::Word("a"),
            Token::Separator(':'),
            Token::Word("i32"),
            Token::Operation('='),
            Token::Number(Ok(Number::AbstractInt(2))),
            Token::Separator(';'),
        ],
    );
}
