macro_rules! bitflags_array_impl {
        ($impl_name:ident $inner_name:ident $name:ident $op:tt $($struct_names:ident)*) => (
            impl core::ops::$impl_name for $name {
                type Output = Self;

                #[inline]
                fn $inner_name(self, other: Self) -> Self {
                    Self {
                        $($struct_names: self.$struct_names $op other.$struct_names,)*
                    }
                }
            }
        )
    }
pub(crate) use bitflags_array_impl;

macro_rules! bitflags_array_impl_assign {
        ($impl_name:ident $inner_name:ident $name:ident $op:tt $($struct_names:ident)*) => (
            impl core::ops::$impl_name for $name {
                #[inline]
                fn $inner_name(&mut self, other: Self) {
                    $(self.$struct_names $op other.$struct_names;)*
                }
            }
        )
    }
pub(crate) use bitflags_array_impl_assign;

macro_rules! bit_array_impl {
        ($impl_name:ident $inner_name:ident $name:ident $op:tt) => (
            impl core::ops::$impl_name for $name {
                type Output = Self;

                #[inline]
                fn $inner_name(mut self, other: Self) -> Self {
                    for (inner, other) in self.0.iter_mut().zip(other.0.iter()) {
                        *inner $op *other;
                    }
                    self
                }
            }
        )
    }
pub(crate) use bit_array_impl;

macro_rules! bitflags_independent_two_arg {
        ($(#[$meta:meta])* $func_name:ident $($struct_names:ident)*) => (
            $(#[$meta])*
            pub const fn $func_name(self, other:Self) -> Self {
                Self { $($struct_names: self.$struct_names.$func_name(other.$struct_names),)* }
            }
        )
    }
pub(crate) use bitflags_independent_two_arg;

// For the most part this macro should not be modified, most configuration should be possible
// without changing this macro.
/// Macro for creating sets of bitflags, we need this because there are almost more flags than bits
/// in a u64, we can't use a u128 because of FFI, and the number of flags is increasing.
///
/// Alternatively we also want to separate WebGPU features from native WGPU features.
macro_rules! bitflags_array {
        (
            $(#[$outer:meta])*
            pub struct ($name:ident, $name_bits:ident): [$T:ty; $Len:expr];

            $(
                $(#[$bit_outer:meta])*
                $vis:vis struct $inner_name:ident $lower_inner_name:ident {
                    $(
                        $(#[doc $($args:tt)*])*
                        #[name($str_name:literal $(, $alias:literal)*)]
                        const $Flag:tt = $value:expr;
                    )*
                }
            )*
        ) => {
            crate::bitflags_array! {
                $(#[$outer])*
                pub struct ($name, $name_bits): [$T; $Len];

                $(
                    $(#[$bit_outer])*
                    $vis struct $inner_name $lower_inner_name {
                        $(
                            $(#[doc $($args)*])*
                            const $Flag = $value;
                        )*
                    }
                )*
            }

            // Parses kebab-case feature names (i.e. the names given in the spec, for features
            // in FeaturesWebGPU, and otherwise the `wgpu-` prefixed names).
            impl FromStr for $name {
                type Err = ();

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    Ok(match s {
                        $($($str_name $(| $alias)* => Self::$Flag,)*)*
                        _ => return Err(()),
                    })
                }
            }

            impl $name {
                /// If the argument is a single [`Features`] flag, returns the corresponding
                /// `kebab-case` feature name, otherwise `None`.
                #[must_use]
                pub fn as_str(&self) -> Option<&'static str> {
                    Some(match *self {
                        $($(Self::$Flag => $str_name,)*)*
                        _ => return None,
                    })
                }
            }
        };
        (
            $(#[$outer:meta])*
            pub struct ($name:ident, $name_bits:ident): [$T:ty; $Len:expr];

            $(
                $(#[$bit_outer:meta])*
                $vis:vis struct $inner_name:ident $lower_inner_name:ident {
                    $(
                        $(#[doc $($args:tt)*])*
                        const $Flag:tt = $value:expr;
                    )*
                }
            )*
        ) => {
            $(
                bitflags::bitflags! {
                    $(#[$bit_outer])*
                    $vis struct $inner_name: $T {
                        $(
                            $(#[doc $($args)*])*
                            const $Flag = $value;
                        )*
                    }
                }
            )*

            $(#[$outer])*
            pub struct $name {
                $(
                    #[allow(missing_docs)]
                    $vis $lower_inner_name: $inner_name,
                )*
            }

            /// Bits from `Features` in array form
            #[derive(Default, Copy, Clone, Debug, PartialEq, Eq)]
            #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
            pub struct $name_bits(pub [$T; $Len]);

            $crate::bitflags_array_impl! { BitOr bitor $name | $($lower_inner_name)* }
            $crate::bitflags_array_impl! { BitAnd bitand $name & $($lower_inner_name)* }
            $crate::bitflags_array_impl! { BitXor bitxor $name ^ $($lower_inner_name)* }
            impl core::ops::Not for $name {
                type Output = Self;

                #[inline]
                fn not(self) -> Self {
                    Self {
                    $($lower_inner_name: !self.$lower_inner_name,)*
                    }
                }
            }
            $crate::bitflags_array_impl! { Sub sub $name - $($lower_inner_name)* }

            #[cfg(feature = "serde")]
            impl Serialize for $name {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: serde::Serializer,
                {
                    bitflags::serde::serialize(self, serializer)
                }
            }

            #[cfg(feature = "serde")]
            impl<'de> Deserialize<'de> for $name {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>,
                {
                    bitflags::serde::deserialize(deserializer)
                }
            }

            impl core::fmt::Display for $name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let mut iter = self.iter_names();
                    // simple look ahead
                    let mut next = iter.next();
                    while let Some((name, _)) = next {
                        f.write_str(name)?;
                        next = iter.next();
                        if next.is_some() {
                            f.write_str(" | ")?;
                        }
                    }
                    Ok(())
                }
            }

            $crate::bitflags_array_impl_assign! { BitOrAssign bitor_assign $name |= $($lower_inner_name)* }
            $crate::bitflags_array_impl_assign! { BitAndAssign bitand_assign $name &= $($lower_inner_name)* }
            $crate::bitflags_array_impl_assign! { BitXorAssign bitxor_assign $name ^= $($lower_inner_name)* }

            $crate::bit_array_impl! { BitOr bitor $name_bits |= }
            $crate::bit_array_impl! { BitAnd bitand $name_bits &= }
            $crate::bit_array_impl! { BitXor bitxor $name_bits ^= }

            impl core::ops::Not for $name_bits {
                type Output = Self;

                #[inline]
                fn not(self) -> Self {
                    let [$($lower_inner_name,)*] = self.0;
                    Self([$(!$lower_inner_name,)*])
                }
            }

            #[cfg(feature = "serde")]
            impl bitflags::parser::WriteHex for $name_bits {
                fn write_hex<W: core::fmt::Write>(&self, mut writer: W) -> core::fmt::Result {
                    let [$($lower_inner_name,)*] = self.0;
                    let mut wrote = false;
                    let mut stager = alloc::string::String::with_capacity(size_of::<$T>() * 2);
                    // we don't want to write it if it's just zero as there may be multiple zeros
                    // resulting in something like "00" being written out. We do want to write it if
                    // there has already been something written though.
                    $(if ($lower_inner_name != 0) || wrote {
                        // First we write to a staging string, then we add any zeros (e.g if #1
                        // is f and a u8 and #2 is a then the two combined would be f0a which requires
                        // a 0 inserted)
                        $lower_inner_name.write_hex(&mut stager)?;
                        if (stager.len() != size_of::<$T>() * 2) && wrote {
                            let zeros_to_write = (size_of::<$T>() * 2) - stager.len();
                            for _ in 0..zeros_to_write {
                                writer.write_char('0')?
                            }
                        }
                        writer.write_str(&stager)?;
                        stager.clear();
                        wrote = true;
                    })*
                    if !wrote {
                        writer.write_str("0")?;
                    }
                    Ok(())
                }
            }

            #[cfg(feature = "serde")]
            impl bitflags::parser::ParseHex for $name_bits {
                fn parse_hex(input: &str) -> Result<Self, bitflags::parser::ParseError> {
                    use bitflags::Bits;

                    let mut unset = Self::EMPTY;
                    let mut end = input.len();
                    if end == 0 {
                        return Err(bitflags::parser::ParseError::empty_flag())
                    }
                    // we iterate starting at the least significant places and going up
                    for (idx, _) in [$(stringify!($lower_inner_name),)*].iter().enumerate().rev() {
                        // A byte is two hex places - u8 (1 byte) = 0x00 (2 hex places).
                        let checked_start = end.checked_sub(size_of::<$T>() * 2);
                        let start = checked_start.unwrap_or(0);

                        let cur_input = &input[start..end];
                        unset.0[idx] = <$T>::from_str_radix(cur_input, 16)
                            .map_err(|_|bitflags::parser::ParseError::invalid_hex_flag(cur_input))?;

                        end = start;

                        if let None = checked_start {
                            break;
                        }
                    }
                    Ok(unset)
                }
            }

            impl bitflags::Bits for $name_bits {
                const EMPTY: Self = $name::empty().bits();

                const ALL: Self = $name::all().bits();
            }

            impl bitflags::Flags for $name {
                const FLAGS: &'static [bitflags::Flag<Self>] = $name::FLAGS;

                type Bits = $name_bits;

                fn bits(&self) -> $name_bits {
                    $name_bits([
                        $(self.$lower_inner_name.bits(),)*
                    ])
                }

                fn from_bits_retain(bits: $name_bits) -> Self {
                    let [$($lower_inner_name,)*] = bits.0;
                    Self {
                        $($lower_inner_name: $inner_name::from_bits_retain($lower_inner_name),)*
                    }
                }

                fn empty() -> Self {
                    Self::empty()
                }

                fn all() -> Self {
                    Self::all()
                }
            }

            impl $name {
                pub(crate) const FLAGS: &'static [bitflags::Flag<Self>] = &[
                    $(
                        $(
                            bitflags::Flag::new(stringify!($Flag), $name::$Flag),
                        )*
                    )*
                ];

                /// Gets the set flags as a container holding an array of bits.
                pub const fn bits(&self) -> $name_bits {
                    $name_bits([
                        $(self.$lower_inner_name.bits(),)*
                    ])
                }

                /// Returns self with no flags set.
                pub const fn empty() -> Self {
                    Self {
                        $($lower_inner_name: $inner_name::empty(),)*
                    }
                }

                /// Returns self with all flags set.
                pub const fn all() -> Self {
                    Self {
                        $($lower_inner_name: $inner_name::all(),)*
                    }
                }

                /// Whether all the bits set in `other` are all set in `self`
                pub const fn contains(self, other:Self) -> bool {
                    // we need an annoying true to catch the last && >:(
                    $(self.$lower_inner_name.contains(other.$lower_inner_name) &&)* true
                }

                /// Returns whether any bit set in `self` matched any bit set in `other`.
                pub const fn intersects(self, other:Self) -> bool {
                    $(self.$lower_inner_name.intersects(other.$lower_inner_name) ||)* false
                }

                /// Returns whether there is no flag set.
                pub const fn is_empty(self) -> bool {
                    $(self.$lower_inner_name.is_empty() &&)* true
                }

                /// Returns whether the struct has all flags set.
                pub const fn is_all(self) -> bool {
                    $(self.$lower_inner_name.is_all() &&)* true
                }

                $crate::bitflags_independent_two_arg! {
                    /// Bitwise or - `self | other`
                    union $($lower_inner_name)*
                }

                $crate::bitflags_independent_two_arg! {
                    /// Bitwise and - `self & other`
                    intersection $($lower_inner_name)*
                }

                $crate::bitflags_independent_two_arg! {
                    /// Bitwise and of the complement of other - `self & !other`
                    difference $($lower_inner_name)*
                }

                $crate::bitflags_independent_two_arg! {
                    /// Bitwise xor - `self ^ other`
                    symmetric_difference $($lower_inner_name)*
                }

                /// Bitwise not - `!self`
                pub const fn complement(self) -> Self {
                    Self {
                        $($lower_inner_name: self.$lower_inner_name.complement(),)*
                    }
                }

                /// Calls [`Self::insert`] if `set` is true and otherwise calls [`Self::remove`].
                pub fn set(&mut self, other:Self, set: bool) {
                    $(self.$lower_inner_name.set(other.$lower_inner_name, set);)*
                }

                /// Inserts specified flag(s) into self
                pub fn insert(&mut self, other:Self) {
                    $(self.$lower_inner_name.insert(other.$lower_inner_name);)*
                }

                /// Removes specified flag(s) from self
                pub fn remove(&mut self, other:Self) {
                    $(self.$lower_inner_name.remove(other.$lower_inner_name);)*
                }

                /// Toggles specified flag(s) in self
                pub fn toggle(&mut self, other:Self) {
                    $(self.$lower_inner_name.toggle(other.$lower_inner_name);)*
                }

                /// Takes in [`FeatureBits`] and returns None if there are invalid bits or otherwise Self with
                /// those bits set
                pub const fn from_bits(bits: $name_bits) -> Option<Self> {
                    let [$($lower_inner_name,)*] = bits.0;
                    // The ? operator does not work in a const context.
                    Some(Self {
                        $(
                            $lower_inner_name: match $inner_name::from_bits($lower_inner_name) {
                                Some(some) => some,
                                None => return None,
                            },
                        )*
                    })
                }

                /// Takes in [`FeatureBits`] and returns Self with only valid bits (all other bits removed)
                pub const fn from_bits_truncate(bits: $name_bits) -> Self {
                    let [$($lower_inner_name,)*] = bits.0;
                    Self { $($lower_inner_name: $inner_name::from_bits_truncate($lower_inner_name),)* }
                }

                /// Takes in [`FeatureBits`] and returns Self with all bits that were set without removing
                /// invalid bits
                pub const fn from_bits_retain(bits: $name_bits) -> Self {
                    let [$($lower_inner_name,)*] = bits.0;
                    Self { $($lower_inner_name: $inner_name::from_bits_retain($lower_inner_name),)* }
                }

                /// Takes in a bitflags flag name (in `SCREAMING_SNAKE_CASE`) and returns Self
                /// if it matches or none if the name does not match the name of any of the
                /// flags. Name is capitalisation dependent.
                ///
                /// [`impl FromStr`] can be used to recognize kebab-case names, like are used in
                /// the WebGPU spec.
                pub fn from_name(name: &str) -> Option<Self> {
                    match name {
                        $(
                            $(
                                stringify!($Flag) => Some(Self::$Flag),
                            )*
                        )*
                        _ => None,
                    }
                }

                /// Combines the features from the internal flags into the entire features struct
                pub fn from_internal_flags($($lower_inner_name: $inner_name,)*) -> Self {
                    Self {
                        $($lower_inner_name,)*
                    }
                }

                /// Returns an iterator over the set flags.
                pub const fn iter(&self) -> bitflags::iter::Iter<$name> {
                    bitflags::iter::Iter::__private_const_new($name::FLAGS, *self, *self)
                }

                /// Returns an iterator over the set flags and their names.
                ///
                /// These are bitflags names in `SCREAMING_SNAKE_CASE`.
                pub const fn iter_names(&self) -> bitflags::iter::IterNames<$name> {
                    bitflags::iter::IterNames::__private_const_new($name::FLAGS, *self, *self)
                }

                $(
                    $(
                        $(#[doc $($args)*])*
                        #[allow(clippy::needless_update, reason = "only useless if there is 1 member")]
                        pub const $Flag: Self = Self {
                            $lower_inner_name: $inner_name::from_bits_truncate($value),
                            ..Self::empty()
                        };
                    )*
                )*
            }

            $(
                impl From<$inner_name> for $name {
                    #[allow(clippy::needless_update, reason = "only useless if there is 1 member")]
                    fn from($lower_inner_name: $inner_name) -> Self {
                        Self {
                            $lower_inner_name,
                            ..Self::empty()
                        }
                    }
                }
            )*
        };
    }
pub(crate) use bitflags_array;
