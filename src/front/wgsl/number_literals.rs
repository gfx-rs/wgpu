use std::convert::TryFrom;

use hexf_parse::parse_hexf32;

use crate::Bytes;

use super::{
    lexer::{try_skip_prefix, Lexer},
    BadFloatError, BadIntError, Error, ExpectedToken, NumberType, Span, Token,
};

fn check_int_literal(word_without_minus: &str, minus: bool, hex: bool) -> Result<(), BadIntError> {
    let leading_zeros = word_without_minus
        .bytes()
        .take_while(|&b| b == b'0')
        .count();

    if word_without_minus == "0" && minus {
        Err(BadIntError::NegativeZero)
    } else if word_without_minus != "0" && !hex && leading_zeros != 0 {
        Err(BadIntError::LeadingZeros)
    } else {
        Ok(())
    }
}

pub fn get_i32_literal(word: &str, span: Span) -> Result<i32, Error<'_>> {
    let (minus, word_without_minus, _) = try_skip_prefix(word, "-");
    let (hex, word_without_minus_and_0x, _) = try_skip_prefix(word_without_minus, "0x");

    check_int_literal(word_without_minus, minus, hex)
        .map_err(|e| Error::BadI32(span.clone(), e))?;

    let parsed_val = match (hex, minus) {
        (true, true) => i32::from_str_radix(&format!("-{}", word_without_minus_and_0x), 16),
        (true, false) => i32::from_str_radix(word_without_minus_and_0x, 16),
        (false, _) => word.parse(),
    };

    parsed_val.map_err(|e| Error::BadI32(span, e.into()))
}

pub fn get_u32_literal(word: &str, span: Span) -> Result<u32, Error<'_>> {
    let (minus, word_without_minus, _) = try_skip_prefix(word, "-");
    let (hex, word_without_minus_and_0x, _) = try_skip_prefix(word_without_minus, "0x");

    check_int_literal(word_without_minus, minus, hex)
        .map_err(|e| Error::BadU32(span.clone(), e))?;

    // We need to add a minus here as well, since the lexer also accepts syntactically incorrect negative uints
    let parsed_val = match (hex, minus) {
        (true, true) => u32::from_str_radix(&format!("-{}", word_without_minus_and_0x), 16),
        (true, false) => u32::from_str_radix(word_without_minus_and_0x, 16),
        (false, _) => word.parse(),
    };

    parsed_val.map_err(|e| Error::BadU32(span, e.into()))
}

pub fn get_f32_literal(word: &str, span: Span) -> Result<f32, Error<'_>> {
    let hex = word.starts_with("0x") || word.starts_with("-0x");

    let parsed_val = if hex {
        parse_hexf32(word, false).map_err(BadFloatError::ParseHexfError)
    } else {
        word.parse::<f32>().map_err(BadFloatError::ParseFloatError)
    };

    parsed_val.map_err(|e| Error::BadFloat(span, e))
}

pub(super) fn _parse_uint_literal<'a>(
    lexer: &mut Lexer<'a>,
    width: Bytes,
) -> Result<u32, Error<'a>> {
    let token_span = lexer.next();

    if width != 4 {
        // Only 32-bit literals supported by the spec and naga for now!
        return Err(Error::BadScalarWidth(token_span.1, width));
    }

    match token_span {
        (
            Token::Number {
                value,
                ty: NumberType::Uint,
                width: token_width,
            },
            span,
        ) if token_width.unwrap_or(4) == width => get_u32_literal(value, span),
        other => Err(Error::Unexpected(
            other,
            ExpectedToken::Number {
                ty: Some(NumberType::Uint),
                width: Some(width),
            },
        )),
    }
}

/// Parse a non-negative signed integer literal.
/// This is for attributes like `size`, `location` and others.
pub(super) fn parse_non_negative_sint_literal<'a>(
    lexer: &mut Lexer<'a>,
    width: Bytes,
) -> Result<u32, Error<'a>> {
    let token_span = lexer.next();

    if width != 4 {
        // Only 32-bit literals supported by the spec and naga for now!
        return Err(Error::BadScalarWidth(token_span.1, width));
    }

    match token_span {
        (
            Token::Number {
                value,
                ty: NumberType::Sint,
                width: token_width,
            },
            span,
        ) if token_width.unwrap_or(4) == width => {
            let i32_val = get_i32_literal(value, span.clone())?;
            u32::try_from(i32_val).map_err(|_| Error::NegativeInt(span))
        }
        other => Err(Error::Unexpected(
            other,
            ExpectedToken::Number {
                ty: Some(NumberType::Sint),
                width: Some(width),
            },
        )),
    }
}

/// Parse a non-negative integer literal that may be either signed or unsigned.
/// This is for the `workgroup_size` attribute and array lengths.
/// Note: these values should be no larger than [`i32::MAX`], but this is not checked here.
pub(super) fn parse_generic_non_negative_int_literal<'a>(
    lexer: &mut Lexer<'a>,
    width: Bytes,
) -> Result<u32, Error<'a>> {
    let token_span = lexer.next();

    if width != 4 {
        // Only 32-bit literals supported by the spec and naga for now!
        return Err(Error::BadScalarWidth(token_span.1, width));
    }

    match token_span {
        (
            Token::Number {
                value,
                ty: NumberType::Sint,
                width: token_width,
            },
            span,
        ) if token_width.unwrap_or(4) == width => {
            let i32_val = get_i32_literal(value, span.clone())?;
            u32::try_from(i32_val).map_err(|_| Error::NegativeInt(span))
        }
        (
            Token::Number {
                value,
                ty: NumberType::Uint,
                width: token_width,
            },
            span,
        ) if token_width.unwrap_or(4) == width => get_u32_literal(value, span),
        other => Err(Error::Unexpected(
            other,
            ExpectedToken::Number {
                ty: Some(NumberType::Sint),
                width: Some(width),
            },
        )),
    }
}

pub(super) fn _parse_float_literal<'a>(
    lexer: &mut Lexer<'a>,
    width: Bytes,
) -> Result<f32, Error<'a>> {
    let token_span = lexer.next();

    if width != 4 {
        // Only 32-bit literals supported by the spec and naga for now!
        return Err(Error::BadScalarWidth(token_span.1, width));
    }

    match token_span {
        (
            Token::Number {
                value,
                ty: NumberType::Float,
                width: token_width,
            },
            span,
        ) if token_width.unwrap_or(4) == width => get_f32_literal(value, span),
        other => Err(Error::Unexpected(
            other,
            ExpectedToken::Number {
                ty: Some(NumberType::Float),
                width: Some(width),
            },
        )),
    }
}
