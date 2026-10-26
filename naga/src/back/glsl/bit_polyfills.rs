//! Polyfills for the integer bit manipulation builtins.
//!
//! `bitfieldExtract`, `bitfieldInsert`, `bitfieldReverse`, `bitCount`,
//! `findLSB` and `findMSB` are only available from GLSL 4.00 and GLSL ES 3.10
//! (see [`Version::supports_integer_functions`]). On older versions (most
//! notably GLSL ES 3.00, which is what WebGL2 uses), the corresponding WGSL
//! builtins are written as calls to helper functions defined here, which only
//! use basic integer arithmetic.
//!
//! The helpers are written once per needed type, before any other function.
//! The core of every helper is implemented for `uint`. The `int` overloads
//! (except for `extractBits`, which needs to sign-extend) reinterpret their
//! arguments as `uint`, and the vector overloads call the scalar overload once
//! per component.
//!
//! Care is taken to never shift by 32 or more, which is undefined in GLSL.
//!
//! [`Version::supports_integer_functions`]: crate::back::glsl::Version::supports_integer_functions

use alloc::{collections::BTreeSet, format, string::String};
use core::fmt::{self, Write};

use crate::{valid, Expression, MathFunction as Mf, Scalar, TypeInner};

/// An integer bit manipulation builtin that has a polyfill.
///
/// The order of the variants matters: helpers are written in the order of
/// [`Helper`]'s `Ord` implementation, so a helper must only depend on helpers
/// of earlier variants (or on earlier overloads of the same variant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BitFunction {
    CountOneBits,
    ReverseBits,
    ExtractBits,
    InsertBits,
    CountTrailingZeros,
    /// Depends on [`BitFunction::CountTrailingZeros`].
    FirstTrailingBit,
    CountLeadingZeros,
    /// Depends on [`BitFunction::CountLeadingZeros`].
    FirstLeadingBit,
}

pub(super) const COUNT_ONE_BITS_FUNCTION: &str = "naga_count_one_bits";
pub(super) const REVERSE_BITS_FUNCTION: &str = "naga_reverse_bits";
pub(super) const EXTRACT_BITS_FUNCTION: &str = "naga_extract_bits";
pub(super) const INSERT_BITS_FUNCTION: &str = "naga_insert_bits";
pub(super) const COUNT_TRAILING_ZEROS_FUNCTION: &str = "naga_count_trailing_zeros";
pub(super) const FIRST_TRAILING_BIT_FUNCTION: &str = "naga_first_trailing_bit";
pub(super) const COUNT_LEADING_ZEROS_FUNCTION: &str = "naga_count_leading_zeros";
pub(super) const FIRST_LEADING_BIT_FUNCTION: &str = "naga_first_leading_bit";

impl BitFunction {
    const fn from_math(fun: Mf) -> Option<Self> {
        Some(match fun {
            Mf::CountOneBits => Self::CountOneBits,
            Mf::ReverseBits => Self::ReverseBits,
            Mf::ExtractBits => Self::ExtractBits,
            Mf::InsertBits => Self::InsertBits,
            Mf::CountTrailingZeros => Self::CountTrailingZeros,
            Mf::FirstTrailingBit => Self::FirstTrailingBit,
            Mf::CountLeadingZeros => Self::CountLeadingZeros,
            Mf::FirstLeadingBit => Self::FirstLeadingBit,
            _ => return None,
        })
    }

    /// The name of the helper function, shared by all of its overloads.
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::CountOneBits => COUNT_ONE_BITS_FUNCTION,
            Self::ReverseBits => REVERSE_BITS_FUNCTION,
            Self::ExtractBits => EXTRACT_BITS_FUNCTION,
            Self::InsertBits => INSERT_BITS_FUNCTION,
            Self::CountTrailingZeros => COUNT_TRAILING_ZEROS_FUNCTION,
            Self::FirstTrailingBit => FIRST_TRAILING_BIT_FUNCTION,
            Self::CountLeadingZeros => COUNT_LEADING_ZEROS_FUNCTION,
            Self::FirstLeadingBit => FIRST_LEADING_BIT_FUNCTION,
        }
    }
}

/// One overload of a helper function.
///
/// Field order matters for the derived `Ord`, see [`BitFunction`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Helper {
    fun: BitFunction,
    /// `false` for `uint`, `true` for `int`.
    signed: bool,
    /// Number of components, 1 for scalars.
    components: u8,
}

impl Helper {
    /// Returns the helper to call for `fun` applied to an argument of type `arg_ty`,
    /// or `None` if `fun` has no polyfill for that type.
    pub(super) fn new(fun: Mf, arg_ty: &TypeInner) -> Option<Self> {
        let fun = BitFunction::from_math(fun)?;
        let (scalar, components) = match *arg_ty {
            TypeInner::Scalar(scalar) => (scalar, 1),
            TypeInner::Vector { size, scalar } => (scalar, size as u8),
            _ => return None,
        };
        let signed = match scalar {
            Scalar::I32 => true,
            Scalar::U32 => false,
            _ => return None,
        };
        Some(Self {
            fun,
            signed,
            components,
        })
    }

    /// The name of the helper function to call.
    pub(super) const fn name(self) -> &'static str {
        self.fun.name()
    }

    /// The other helpers this one calls.
    fn dependencies(self) -> impl Iterator<Item = Self> {
        let scalar = Self {
            components: 1,
            ..self
        };
        let unsigned_scalar = Self {
            signed: false,
            ..scalar
        };
        let component_wise = (self.components > 1).then_some(scalar);
        let reinterpreted =
            (self.components == 1 && self.signed && self.fun != BitFunction::ExtractBits)
                .then_some(unsigned_scalar);
        let other = if self == unsigned_scalar {
            match self.fun {
                BitFunction::FirstTrailingBit => Some(Self {
                    fun: BitFunction::CountTrailingZeros,
                    ..self
                }),
                BitFunction::FirstLeadingBit => Some(Self {
                    fun: BitFunction::CountLeadingZeros,
                    ..self
                }),
                _ => None,
            }
        } else {
            None
        };
        component_wise.into_iter().chain(reinterpreted).chain(other)
    }

    fn insert_with_dependencies(self, set: &mut BTreeSet<Self>) {
        if set.insert(self) {
            for dependency in self.dependencies() {
                dependency.insert_with_dependencies(set);
            }
        }
    }

    /// Writes the definition of this overload of the helper function.
    ///
    /// The overloads it depends on must have been written already.
    fn write(self, out: &mut impl Write) -> fmt::Result {
        let name = self.fun.name();
        let ty = match (self.signed, self.components) {
            (false, 1) => "uint".into(),
            (true, 1) => "int".into(),
            (false, n) => format!("uvec{n}"),
            (true, n) => format!("ivec{n}"),
        };

        let params = match self.fun {
            BitFunction::ExtractBits => format!("{ty} e, uint offset, uint count"),
            BitFunction::InsertBits => format!("{ty} e, {ty} newbits, uint offset, uint count"),
            _ => format!("{ty} e"),
        };

        writeln!(out)?;
        writeln!(out, "{ty} {name}({params}) {{")?;

        if self.components > 1 {
            // Call the scalar overload for each component.
            write!(out, "    return {ty}(")?;
            for (i, c) in ["x", "y", "z", "w"]
                .into_iter()
                .take(self.components as usize)
                .enumerate()
            {
                if i != 0 {
                    write!(out, ", ")?;
                }
                match self.fun {
                    BitFunction::ExtractBits => write!(out, "{name}(e.{c}, offset, count)")?,
                    BitFunction::InsertBits => {
                        write!(out, "{name}(e.{c}, newbits.{c}, offset, count)")?
                    }
                    _ => write!(out, "{name}(e.{c})")?,
                }
            }
            writeln!(out, ");")?;
            return writeln!(out, "}}");
        }

        let body: String = match (self.fun, self.signed) {
            (BitFunction::CountOneBits, false) => "    uint v = e - ((e >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24u;"
                .into(),
            (BitFunction::ReverseBits, false) => "    uint v = e;
    v = ((v >> 1u) & 0x55555555u) | ((v & 0x55555555u) << 1u);
    v = ((v >> 2u) & 0x33333333u) | ((v & 0x33333333u) << 2u);
    v = ((v >> 4u) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4u);
    v = ((v >> 8u) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8u);
    return (v >> 16u) | (v << 16u);"
                .into(),
            (BitFunction::ExtractBits, false) => "    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return 0u;
    }
    if (c == 32u) {
        return e;
    }
    return (e >> o) & ((1u << c) - 1u);"
                .into(),
            (BitFunction::ExtractBits, true) => {
                // Move the field to the most significant bits, then use an arithmetic
                // right shift to move it back down while sign-extending it.
                "    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return 0;
    }
    if (c == 32u) {
        return e;
    }
    return int(uint(e) << (32u - o - c)) >> (32u - c);"
                    .into()
            }
            (BitFunction::InsertBits, false) => "    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return e;
    }
    if (c == 32u) {
        return newbits;
    }
    uint mask = ((1u << c) - 1u) << o;
    return (e & ~mask) | ((newbits << o) & mask);"
                .into(),
            (BitFunction::CountTrailingZeros, false) => "    if (e == 0u) {
        return 32u;
    }
    uint v = e;
    uint n = 0u;
    if ((v & 0x0000FFFFu) == 0u) { n += 16u; v >>= 16u; }
    if ((v & 0x000000FFu) == 0u) { n += 8u; v >>= 8u; }
    if ((v & 0x0000000Fu) == 0u) { n += 4u; v >>= 4u; }
    if ((v & 0x00000003u) == 0u) { n += 2u; v >>= 2u; }
    if ((v & 0x00000001u) == 0u) { n += 1u; }
    return n;"
                .into(),
            (BitFunction::FirstTrailingBit, false) => {
                format!("    return e == 0u ? 0xFFFFFFFFu : {COUNT_TRAILING_ZEROS_FUNCTION}(e);")
            }
            (BitFunction::CountLeadingZeros, false) => "    if (e == 0u) {
        return 32u;
    }
    uint v = e;
    uint n = 0u;
    if ((v & 0xFFFF0000u) == 0u) { n += 16u; v <<= 16u; }
    if ((v & 0xFF000000u) == 0u) { n += 8u; v <<= 8u; }
    if ((v & 0xF0000000u) == 0u) { n += 4u; v <<= 4u; }
    if ((v & 0xC0000000u) == 0u) { n += 2u; v <<= 2u; }
    if ((v & 0x80000000u) == 0u) { n += 1u; }
    return n;"
                .into(),
            (BitFunction::FirstLeadingBit, false) => {
                format!(
                    "    return e == 0u ? 0xFFFFFFFFu : 31u - {COUNT_LEADING_ZEROS_FUNCTION}(e);"
                )
            }
            (BitFunction::FirstLeadingBit, true) => {
                // For negative numbers, look for the most significant 0 bit instead.
                format!("    return int({name}(uint(e < 0 ? ~e : e)));")
            }
            (BitFunction::InsertBits, true) => {
                format!("    return int({name}(uint(e), uint(newbits), offset, count));")
            }
            (_, true) => {
                // The result has the same bit pattern as for the unsigned argument.
                format!("    return int({name}(uint(e)));")
            }
        };
        writeln!(out, "{body}")?;
        writeln!(out, "}}")
    }
}

/// Collects the helpers needed by `function`, including their dependencies.
pub(super) fn collect_needed(
    function: &crate::Function,
    info: &valid::FunctionInfo,
    types: &crate::UniqueArena<crate::Type>,
    needed: &mut BTreeSet<Helper>,
) {
    for (_, expression) in function.expressions.iter() {
        if let Expression::Math { fun, arg, .. } = *expression {
            if let Some(helper) = Helper::new(fun, info[arg].ty.inner_with(types)) {
                helper.insert_with_dependencies(needed);
            }
        }
    }
}

/// Writes the definitions of all `helpers`, in dependency order.
pub(super) fn write_helpers(out: &mut impl Write, helpers: &BTreeSet<Helper>) -> fmt::Result {
    for helper in helpers {
        helper.write(out)?;
    }
    Ok(())
}
