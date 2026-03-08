use alloc::format;

use crate::front::wgsl::error::NumberError;
use crate::front::wgsl::parse::directive::enable_extension::ImplementedEnableExtension;
use crate::front::wgsl::parse::lexer::Token;
use half::f16;

/// When using this type assume no Abstract Int/Float for now
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Number {
    /// Abstract Int (-2^63 ≤ i < 2^63)
    AbstractInt(i64),
    /// Abstract Float (IEEE-754 binary64)
    AbstractFloat(f64),
    /// Concrete i32
    I32(i32),
    /// Concrete u32
    U32(u32),
    /// Concrete i64
    I64(i64),
    /// Concrete u64
    U64(u64),
    /// Concrete f16
    F16(f16),
    /// Concrete f32
    F32(f32),
    /// Concrete f64
    F64(f64),
}

impl Number {
    pub(super) const fn requires_enable_extension(&self) -> Option<ImplementedEnableExtension> {
        match *self {
            Number::F16(_) => Some(ImplementedEnableExtension::F16),
            _ => None,
        }
    }
}

pub(in crate::front::wgsl) fn consume_number(input: &str) -> (Token<'_>, &str) {
    let (result, rest) = parse(input);
    (Token::Number(result), rest)
}

enum Kind {
    Int(IntKind),
    Float(FloatKind),
}

enum IntKind {
    I32,
    U32,
    I64,
    U64,
}

#[derive(Debug)]
enum FloatKind {
    F16,
    F32,
    F64,
}

// The following regexes (from the WGSL spec) will be matched:

// int_literal:
// | / 0                                                                [iu]?   /
// | / [1-9][0-9]*                                                      [iu]?   /
// | / 0[xX][0-9a-fA-F]+                                                [iu]?   /

// decimal_float_literal:
// | / 0                                                                [fh]    /
// | / [1-9][0-9]*                                                      [fh]    /
// | / [0-9]*               \.[0-9]+            ([eE][+-]?[0-9]+)?      [fh]?   /
// | / [0-9]+               \.[0-9]*            ([eE][+-]?[0-9]+)?      [fh]?   /
// | / [0-9]+                                    [eE][+-]?[0-9]+        [fh]?   /

// hex_float_literal:
// | / 0[xX][0-9a-fA-F]*    \.[0-9a-fA-F]+      ([pP][+-]?[0-9]+        [fh]?)? /
// | / 0[xX][0-9a-fA-F]+    \.[0-9a-fA-F]*      ([pP][+-]?[0-9]+        [fh]?)? /
// | / 0[xX][0-9a-fA-F]+                         [pP][+-]?[0-9]+        [fh]?   /

// You could visualize the regex below via https://debuggex.com to get a rough idea what `parse` is doing
// (?:0[xX](?:([0-9a-fA-F]+\.[0-9a-fA-F]*|[0-9a-fA-F]*\.[0-9a-fA-F]+)(?:([pP][+-]?[0-9]+)([fh]?))?|([0-9a-fA-F]+)([pP][+-]?[0-9]+)([fh]?)|([0-9a-fA-F]+)([iu]?))|((?:[0-9]+[eE][+-]?[0-9]+|(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][+-]?[0-9]+)?))([fh]?)|((?:[0-9]|[1-9][0-9]+))([iufh]?))

// Leading signs are handled as unary operators.

fn parse(input: &str) -> (Result<Number, NumberError>, &str) {
    /// returns `true` and consumes `X` bytes from the given byte buffer
    /// if the given `X` nr of patterns are found at the start of the buffer
    macro_rules! consume {
        ($bytes:ident, $($pattern:pat),*) => {
            match $bytes {
                &[$($pattern),*, ref rest @ ..] => { $bytes = rest; true },
                _ => false,
            }
        };
    }

    /// consumes one byte from the given byte buffer
    /// if one of the given patterns are found at the start of the buffer
    /// returning the corresponding expr for the matched pattern
    macro_rules! consume_map {
        ($bytes:ident, [$( $($pattern:pat_param),* => $to:expr),* $(,)?]) => {
            match $bytes {
                $( &[ $($pattern),*, ref rest @ ..] => { $bytes = rest; Some($to) }, )*
                _ => None,
            }
        };
    }

    /// consumes all consecutive bytes matched by the `0-9` pattern from the given byte buffer
    /// returning the number of consumed bytes
    macro_rules! consume_dec_digits {
        ($bytes:ident) => {{
            let start_len = $bytes.len();
            while let &[b'0'..=b'9', ref rest @ ..] = $bytes {
                $bytes = rest;
            }
            start_len - $bytes.len()
        }};
    }

    /// consumes all consecutive bytes matched by the `0-9 | a-f | A-F` pattern from the given byte buffer
    /// returning the number of consumed bytes
    macro_rules! consume_hex_digits {
        ($bytes:ident) => {{
            let start_len = $bytes.len();
            while let &[b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F', ref rest @ ..] = $bytes {
                $bytes = rest;
            }
            start_len - $bytes.len()
        }};
    }

    macro_rules! consume_float_suffix {
        ($bytes:ident) => {
            consume_map!($bytes, [
                b'h' => FloatKind::F16,
                b'f' => FloatKind::F32,
                b'l', b'f' => FloatKind::F64,
            ])
        };
    }

    /// maps the given `&[u8]` (tail of the initial `input: &str`) to a `&str`
    macro_rules! rest_to_str {
        ($bytes:ident) => {
            &input[input.len() - $bytes.len()..]
        };
    }

    struct ExtractSubStr<'a>(&'a str);

    impl<'a> ExtractSubStr<'a> {
        /// given an `input` and a `start` (tail of the `input`)
        /// creates a new [`ExtractSubStr`](`Self`)
        fn start(input: &'a str, start: &'a [u8]) -> Self {
            let start = input.len() - start.len();
            Self(&input[start..])
        }
        /// given an `end` (tail of the initial `input`)
        /// returns a substring of `input`
        fn end(&self, end: &'a [u8]) -> &'a str {
            let end = self.0.len() - end.len();
            &self.0[..end]
        }
    }

    let mut bytes = input.as_bytes();

    let general_extract = ExtractSubStr::start(input, bytes);

    if consume!(bytes, b'0', b'x' | b'X') {
        let digits_extract = ExtractSubStr::start(input, bytes);

        let consumed = consume_hex_digits!(bytes);

        if consume!(bytes, b'.') {
            let consumed_after_period = consume_hex_digits!(bytes);

            if consumed + consumed_after_period == 0 {
                return (Err(NumberError::Invalid), rest_to_str!(bytes));
            }

            let significand = general_extract.end(bytes);

            if consume!(bytes, b'p' | b'P') {
                consume!(bytes, b'+' | b'-');
                let consumed = consume_dec_digits!(bytes);

                if consumed == 0 {
                    return (Err(NumberError::Invalid), rest_to_str!(bytes));
                }

                let number = general_extract.end(bytes);

                let kind = consume_float_suffix!(bytes);

                (parse_hex_float(number, kind), rest_to_str!(bytes))
            } else {
                (
                    parse_hex_float_missing_exponent(significand, None),
                    rest_to_str!(bytes),
                )
            }
        } else {
            if consumed == 0 {
                return (Err(NumberError::Invalid), rest_to_str!(bytes));
            }

            let significand = general_extract.end(bytes);
            let digits = digits_extract.end(bytes);

            let exp_extract = ExtractSubStr::start(input, bytes);

            if consume!(bytes, b'p' | b'P') {
                consume!(bytes, b'+' | b'-');
                let consumed = consume_dec_digits!(bytes);

                if consumed == 0 {
                    return (Err(NumberError::Invalid), rest_to_str!(bytes));
                }

                let exponent = exp_extract.end(bytes);

                let kind = consume_float_suffix!(bytes);

                (
                    parse_hex_float_missing_period(significand, exponent, kind),
                    rest_to_str!(bytes),
                )
            } else {
                let kind = consume_map!(bytes, [
                    b'i' => IntKind::I32,
                    b'u' => IntKind::U32,
                    b'l', b'i' => IntKind::I64,
                    b'l', b'u' => IntKind::U64,
                ]);

                (parse_hex_int(digits, kind), rest_to_str!(bytes))
            }
        }
    } else {
        let is_first_zero = bytes.first() == Some(&b'0');

        let consumed = consume_dec_digits!(bytes);

        if consume!(bytes, b'.') {
            let consumed_after_period = consume_dec_digits!(bytes);

            if consumed + consumed_after_period == 0 {
                return (Err(NumberError::Invalid), rest_to_str!(bytes));
            }

            if consume!(bytes, b'e' | b'E') {
                consume!(bytes, b'+' | b'-');
                let consumed = consume_dec_digits!(bytes);

                if consumed == 0 {
                    return (Err(NumberError::Invalid), rest_to_str!(bytes));
                }
            }

            let number = general_extract.end(bytes);

            let kind = consume_float_suffix!(bytes);

            (parse_dec_float(number, kind), rest_to_str!(bytes))
        } else {
            if consumed == 0 {
                return (Err(NumberError::Invalid), rest_to_str!(bytes));
            }

            if consume!(bytes, b'e' | b'E') {
                consume!(bytes, b'+' | b'-');
                let consumed = consume_dec_digits!(bytes);

                if consumed == 0 {
                    return (Err(NumberError::Invalid), rest_to_str!(bytes));
                }

                let number = general_extract.end(bytes);

                let kind = consume_float_suffix!(bytes);

                (parse_dec_float(number, kind), rest_to_str!(bytes))
            } else {
                // make sure the multi-digit numbers don't start with zero
                if consumed > 1 && is_first_zero {
                    return (Err(NumberError::Invalid), rest_to_str!(bytes));
                }

                let digits = general_extract.end(bytes);

                let kind = consume_map!(bytes, [
                    b'i' => Kind::Int(IntKind::I32),
                    b'u' => Kind::Int(IntKind::U32),
                    b'l', b'i' => Kind::Int(IntKind::I64),
                    b'l', b'u' => Kind::Int(IntKind::U64),
                    b'h' => Kind::Float(FloatKind::F16),
                    b'f' => Kind::Float(FloatKind::F32),
                    b'l', b'f' => Kind::Float(FloatKind::F64),
                ]);

                (parse_dec(digits, kind), rest_to_str!(bytes))
            }
        }
    }
}

fn parse_hex_float_missing_exponent(
    // format: 0[xX] ( [0-9a-fA-F]+\.[0-9a-fA-F]* | [0-9a-fA-F]*\.[0-9a-fA-F]+ )
    significand: &str,
    kind: Option<FloatKind>,
) -> Result<Number, NumberError> {
    let hexf_input = format!("{}{}", significand, "p0");
    parse_hex_float(&hexf_input, kind)
}

fn parse_hex_float_missing_period(
    // format: 0[xX] [0-9a-fA-F]+
    significand: &str,
    // format: [pP][+-]?[0-9]+
    exponent: &str,
    kind: Option<FloatKind>,
) -> Result<Number, NumberError> {
    let hexf_input = format!("{significand}.{exponent}");
    parse_hex_float(&hexf_input, kind)
}

fn parse_hex_int(
    // format: [0-9a-fA-F]+
    digits: &str,
    kind: Option<IntKind>,
) -> Result<Number, NumberError> {
    parse_int(digits, kind, 16)
}

fn parse_dec(
    // format: ( [0-9] | [1-9][0-9]+ )
    digits: &str,
    kind: Option<Kind>,
) -> Result<Number, NumberError> {
    match kind {
        None => parse_int(digits, None, 10),
        Some(Kind::Int(kind)) => parse_int(digits, Some(kind), 10),
        Some(Kind::Float(kind)) => parse_dec_float(digits, Some(kind)),
    }
}

// Float parsing notes

// The following chapters of IEEE 754-2019 are relevant:
//
// 7.4 Overflow (largest finite number is exceeded by what would have been
//     the rounded floating-point result were the exponent range unbounded)
//
// 7.5 Underflow (tiny non-zero result is detected;
//     for decimal formats tininess is detected before rounding when a non-zero result
//     computed as though both the exponent range and the precision were unbounded
//     would lie strictly between 2^−126)
//
// 7.6 Inexact (rounded result differs from what would have been computed
//     were both exponent range and precision unbounded)

// The WGSL spec requires us to error:
//   on overflow for decimal floating point literals
//   on overflow and inexact for hexadecimal floating point literals
// (underflow is not mentioned)

// rust std lib float from str handles overflow, underflow, inexact transparently (rounds and will not error)

// Therefore we only check for overflow manually for decimal floating point literals

// input format: 0[xX] ( [0-9a-fA-F]+\.[0-9a-fA-F]* | [0-9a-fA-F]*\.[0-9a-fA-F]+ ) [pP][+-]?[0-9]+
fn parse_hex_float(input: &str, kind: Option<FloatKind>) -> Result<Number, NumberError> {
    match kind {
        None => {
            let (neg, mant, exp) = parse_hex_float_parts(input.as_bytes())?;
            let bits = convert_hex_float(neg, mant, exp, F64)?;
            let num = f64::from_bits(bits);

            Ok(Number::AbstractFloat(num))
        }
        // TODO: f16 is not supported
        Some(FloatKind::F16) => Err(NumberError::NotRepresentable),
        Some(FloatKind::F32) => {
            let (neg, mant, exp) = parse_hex_float_parts(input.as_bytes())?;
            let bits = convert_hex_float(neg, mant, exp, F32)?;
            let num = f32::from_bits(bits as u32);

            Ok(Number::F32(num))
        }
        Some(FloatKind::F64) => {
            let (neg, mant, exp) = parse_hex_float_parts(input.as_bytes())?;
            let bits = convert_hex_float(neg, mant, exp, F64)?;
            let num = f64::from_bits(bits);

            Ok(Number::F64(num))
        }
    }
}

// a config for representing a hexadecimal floating-point
struct HexFloatFormat {
    mant_bits: usize,  // number of bits in the mantissa (excluding implicit leading 1)
    precision: usize,  // total precision in bits including implicit bit
    bias: i32,         // exponent bias
    max_exp: i32,      // max exponent before overflow
    exp_bits: usize,   // number of bits in exponent
    min_norm_exp: i32, // smallest exponent for normalized numbers
}

const F32: HexFloatFormat = HexFloatFormat {
    mant_bits: 23,
    precision: 24,
    bias: 127,
    max_exp: 127,
    exp_bits: 8,
    min_norm_exp: -126,
};

const F64: HexFloatFormat = HexFloatFormat {
    mant_bits: 52,
    precision: 53,
    bias: 1023,
    max_exp: 1023,
    exp_bits: 11,
    min_norm_exp: -1022,
};

// derived from hexf-parse module: https://github.com/lifthrasiir/hexf (0BSD)
// parses a hexadecimal floating-point string into its sign, mantissa, and exponent
// input format: 0[xX] ( [0-9a-fA-F]+\.[0-9a-fA-F]* | [0-9a-fA-F]*\.[0-9a-fA-F]+ ) [pP][+-]?[0-9]+
fn parse_hex_float_parts(s: &[u8]) -> Result<(bool, u64, i32), NumberError> {
    let (s, negative) = match s.split_first() {
        Some((&b'+', s)) => (s, false),
        Some((&b'-', s)) => (s, true),
        Some(_) => (s, false),
        // empty
        None => return Err(NumberError::Invalid),
    };

    if !(s.starts_with(b"0x") || s.starts_with(b"0X")) {
        return Err(NumberError::Invalid);
    }

    let mut s = &s[2..];
    let mut acc: u128 = 0;
    let mut digit_seen = false;

    // integer part: [0-9a-fA-F]+
    loop {
        let (rest, digit) = match s.split_first() {
            Some((&c @ b'0'..=b'9', s)) => (s, c - b'0'),
            Some((&c @ b'a'..=b'f', s)) => (s, c - b'a' + 10),
            Some((&c @ b'A'..=b'F', s)) => (s, c - b'A' + 10),
            _ => break,
        };
        s = rest;
        digit_seen = true;
        acc = acc.checked_shl(4).ok_or(NumberError::NotRepresentable)? | digit as u128;
    }

    // fractional part: \.[0-9a-fA-F]+
    let mut nfracs: i32 = 0;
    let mut frac_digit_seen = false;
    if s.starts_with(b".") {
        s = &s[1..];
        loop {
            let (rest, digit) = match s.split_first() {
                Some((&c @ b'0'..=b'9', s)) => (s, c - b'0'),
                Some((&c @ b'a'..=b'f', s)) => (s, c - b'a' + 10),
                Some((&c @ b'A'..=b'F', s)) => (s, c - b'A' + 10),
                _ => break,
            };
            s = rest;
            frac_digit_seen = true;
            acc = acc.checked_shl(4).ok_or(NumberError::NotRepresentable)? | digit as u128;
            nfracs = nfracs.checked_add(1).ok_or(NumberError::NotRepresentable)?;
        }
    }

    if !(digit_seen || frac_digit_seen) {
        return Err(NumberError::Invalid);
    }

    // exponent marker 'p' or 'P'
    let s = match s.split_first() {
        Some((&b'P', s)) | Some((&b'p', s)) => s,
        _ => return Err(NumberError::Invalid),
    };

    // exponent sign
    let (mut s, negative_exponent) = match s.split_first() {
        Some((&b'+', s)) => (s, false),
        Some((&b'-', s)) => (s, true),
        Some(_) => (s, false),
        None => return Err(NumberError::Invalid),
    };

    // exponent digits: [0-9]+
    let mut digit_seen = false;
    let mut exponent: i32 = 0;
    loop {
        let (rest, digit) = match s.split_first() {
            Some((&c @ b'0'..=b'9', s)) => (s, c - b'0'),
            None if digit_seen => break,
            _ => return Err(NumberError::Invalid),
        };
        s = rest;
        digit_seen = true;

        // only update exponent if non‑zero mantissa
        if acc != 0 {
            exponent = exponent
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit as i32))
                .ok_or(NumberError::NotRepresentable)?;
        }
    }

    if negative_exponent {
        exponent = -exponent;
    }

    if acc == 0 {
        return Ok((negative, 0, 0));
    }

    // adjust exponent by 4 per fractional digit
    let exp_adj = nfracs.checked_mul(4).ok_or(NumberError::NotRepresentable)?;
    let exponent = exponent
        .checked_sub(exp_adj)
        .ok_or(NumberError::NotRepresentable)?;

    // remove trailing hex zeros
    let mut mant = acc;
    let mut extra_shift = 0i32;
    while mant > 0 && (mant & 0xF) == 0 {
        mant >>= 4;
        extra_shift = extra_shift
            .checked_add(4)
            .ok_or(NumberError::NotRepresentable)?;
    }

    // final mantissa must fit in 64 bits
    if mant > u64::MAX as u128 {
        return Err(NumberError::NotRepresentable);
    }

    let exponent = exponent
        .checked_add(extra_shift)
        .ok_or(NumberError::NotRepresentable)?;

    Ok((negative, mant as u64, exponent))
}

fn convert_hex_float(
    negative: bool,
    mant: u64,
    exp: i32,
    fmt: HexFloatFormat,
) -> Result<u64, NumberError> {
    let sign_shift = fmt.mant_bits + fmt.exp_bits;
    let sign = (negative as u64) << sign_shift;

    if mant == 0 {
        return Ok(sign);
    }

    let k = 63usize - mant.leading_zeros() as usize;
    let normalexp = exp
        .checked_add(k as i32)
        .ok_or(NumberError::NotRepresentable)?;

    if normalexp > fmt.max_exp {
        return Err(NumberError::NotRepresentable);
    }

    // shift to align mantissa
    let shift = k as i32 - ((fmt.precision as i32) - 1);
    let mut mant_field: u64;

    if normalexp >= fmt.min_norm_exp {
        // normalized
        if shift > 0 {
            if shift >= 64 || (mant & ((1u64 << shift) - 1)) != 0 {
                return Err(NumberError::NotRepresentable);
            }
            mant_field = mant >> shift;
        } else {
            mant_field = mant << -shift;
        }

        mant_field &= (1u64 << fmt.mant_bits) - 1;
        let expo_field = (normalexp + fmt.bias) as u64;

        Ok(sign | (expo_field << fmt.mant_bits) | mant_field)
    } else {
        // subnormal
        let shift_sub = exp - (fmt.min_norm_exp - ((fmt.precision as i32) - 1));
        if shift_sub < 0 {
            let rs = (-shift_sub) as usize;
            if rs >= 64 || (mant & ((1u64 << rs) - 1)) != 0 {
                return Err(NumberError::NotRepresentable);
            }
            mant_field = mant >> rs;
        } else {
            mant_field = mant << shift_sub as u32;
            if (mant_field >> fmt.mant_bits) != 0 {
                return Err(NumberError::NotRepresentable);
            }
        }

        if mant_field == 0 {
            return Err(NumberError::NotRepresentable);
        }

        Ok(sign | (mant_field & ((1u64 << fmt.mant_bits) - 1)))
    }
}

// input format: ( [0-9]+\.[0-9]* | [0-9]*\.[0-9]+ ) ([eE][+-]?[0-9]+)?
//             | [0-9]+ [eE][+-]?[0-9]+
fn parse_dec_float(input: &str, kind: Option<FloatKind>) -> Result<Number, NumberError> {
    match kind {
        None => {
            let num = input.parse::<f64>().unwrap(); // will never fail
            num.is_finite()
                .then_some(Number::AbstractFloat(num))
                .ok_or(NumberError::NotRepresentable)
        }
        Some(FloatKind::F32) => {
            let num = input.parse::<f32>().unwrap(); // will never fail
            num.is_finite()
                .then_some(Number::F32(num))
                .ok_or(NumberError::NotRepresentable)
        }
        Some(FloatKind::F64) => {
            let num = input.parse::<f64>().unwrap(); // will never fail
            num.is_finite()
                .then_some(Number::F64(num))
                .ok_or(NumberError::NotRepresentable)
        }
        Some(FloatKind::F16) => {
            let num = input.parse::<f16>().unwrap(); // will never fail
            num.is_finite()
                .then_some(Number::F16(num))
                .ok_or(NumberError::NotRepresentable)
        }
    }
}

fn parse_int(input: &str, kind: Option<IntKind>, radix: u32) -> Result<Number, NumberError> {
    fn map_err(e: core::num::ParseIntError) -> NumberError {
        match *e.kind() {
            core::num::IntErrorKind::PosOverflow | core::num::IntErrorKind::NegOverflow => {
                NumberError::NotRepresentable
            }
            _ => unreachable!(),
        }
    }
    match kind {
        None => match i64::from_str_radix(input, radix) {
            Ok(num) => Ok(Number::AbstractInt(num)),
            Err(e) => Err(map_err(e)),
        },
        Some(IntKind::I32) => match i32::from_str_radix(input, radix) {
            Ok(num) => Ok(Number::I32(num)),
            Err(e) => Err(map_err(e)),
        },
        Some(IntKind::U32) => match u32::from_str_radix(input, radix) {
            Ok(num) => Ok(Number::U32(num)),
            Err(e) => Err(map_err(e)),
        },
        Some(IntKind::I64) => match i64::from_str_radix(input, radix) {
            Ok(num) => Ok(Number::I64(num)),
            Err(e) => Err(map_err(e)),
        },
        Some(IntKind::U64) => match u64::from_str_radix(input, radix) {
            Ok(num) => Ok(Number::U64(num)),
            Err(e) => Err(map_err(e)),
        },
    }
}
