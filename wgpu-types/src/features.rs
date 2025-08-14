use crate::VertexFormat;
#[cfg(feature = "serde")]
use alloc::fmt;
use alloc::vec::Vec;
#[cfg(feature = "serde")]
use bitflags::parser::{ParseError, ParseHex, WriteHex};
#[cfg(feature = "serde")]
use bitflags::Bits;
use bitflags::Flags;
#[cfg(feature = "serde")]
use core::mem::size_of;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use webgpu_impl::*;
mod webgpu_impl {
    //! Constant values for [`super::FeaturesWebGPU`], separated so they can be picked up by
    //! `cbindgen` in `mozilla-central` (where Firefox is developed).
    #![allow(missing_docs)]

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_DEPTH_CLIP_CONTROL: u64 = 1 << 0;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_DEPTH32FLOAT_STENCIL8: u64 = 1 << 1;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TEXTURE_COMPRESSION_BC: u64 = 1 << 2;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TEXTURE_COMPRESSION_BC_SLICED_3D: u64 = 1 << 3;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TEXTURE_COMPRESSION_ETC2: u64 = 1 << 4;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TEXTURE_COMPRESSION_ASTC: u64 = 1 << 5;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TEXTURE_COMPRESSION_ASTC_SLICED_3D: u64 = 1 << 6;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_TIMESTAMP_QUERY: u64 = 1 << 7;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_INDIRECT_FIRST_INSTANCE: u64 = 1 << 8;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_SHADER_F16: u64 = 1 << 9;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_RG11B10UFLOAT_RENDERABLE: u64 = 1 << 10;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_BGRA8UNORM_STORAGE: u64 = 1 << 11;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_FLOAT32_FILTERABLE: u64 = 1 << 12;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_DUAL_SOURCE_BLENDING: u64 = 1 << 13;

    #[doc(hidden)]
    pub const WEBGPU_FEATURE_CLIP_DISTANCES: u64 = 1 << 14;
}

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

macro_rules! bitflags_independent_two_arg {
    ($(#[$meta:meta])* $func_name:ident $($struct_names:ident)*) => (
        $(#[$meta])*
        pub const fn $func_name(self, other:Self) -> Self {
            Self { $($struct_names: self.$struct_names.$func_name(other.$struct_names),)* }
        }
    )
}

// For the most part this macro should not be modified, most configuration should be possible
// without changing this macro.
/// Macro for creating sets of bitflags, we need this because there are almost more flags than bits
/// in a u64, we can't use a u128 because of FFI, and the number of flags is increasing.
macro_rules! bitflags_array {
    (
        $(#[$outer:meta])*
        pub struct $name:ident: [$T:ty; $Len:expr];

        $(
            $(#[$bit_outer:meta])*
            $vis:vis struct $inner_name:ident $lower_inner_name:ident {
                $(
                    $(#[$inner:ident $($args:tt)*])*
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
                        $(#[$inner $($args)*])*
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
        pub struct FeatureBits(pub [$T; $Len]);

        bitflags_array_impl! { BitOr bitor $name | $($lower_inner_name)* }
        bitflags_array_impl! { BitAnd bitand $name & $($lower_inner_name)* }
        bitflags_array_impl! { BitXor bitxor $name ^ $($lower_inner_name)* }
        impl core::ops::Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self {
                   $($lower_inner_name: !self.$lower_inner_name,)*
                }
            }
        }
        bitflags_array_impl! { Sub sub $name - $($lower_inner_name)* }

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

        bitflags_array_impl_assign! { BitOrAssign bitor_assign $name |= $($lower_inner_name)* }
        bitflags_array_impl_assign! { BitAndAssign bitand_assign $name &= $($lower_inner_name)* }
        bitflags_array_impl_assign! { BitXorAssign bitxor_assign $name ^= $($lower_inner_name)* }

        bit_array_impl! { BitOr bitor FeatureBits |= }
        bit_array_impl! { BitAnd bitand FeatureBits &= }
        bit_array_impl! { BitXor bitxor FeatureBits ^= }

        impl core::ops::Not for FeatureBits {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                let [$($lower_inner_name,)*] = self.0;
                Self([$(!$lower_inner_name,)*])
            }
        }

        #[cfg(feature = "serde")]
        impl WriteHex for FeatureBits {
            fn write_hex<W: fmt::Write>(&self, mut writer: W) -> fmt::Result {
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
        impl ParseHex for FeatureBits {
            fn parse_hex(input: &str) -> Result<Self, ParseError> {

                let mut unset = Self::EMPTY;
                let mut end = input.len();
                if end == 0 {
                    return Err(ParseError::empty_flag())
                }
                // we iterate starting at the least significant places and going up
                for (idx, _) in [$(stringify!($lower_inner_name),)*].iter().enumerate().rev() {
                    // A byte is two hex places - u8 (1 byte) = 0x00 (2 hex places).
                    let checked_start = end.checked_sub(size_of::<$T>() * 2);
                    let start = checked_start.unwrap_or(0);

                    let cur_input = &input[start..end];
                    unset.0[idx] = <$T>::from_str_radix(cur_input, 16)
                        .map_err(|_|ParseError::invalid_hex_flag(cur_input))?;

                    end = start;

                    if let None = checked_start {
                        break;
                    }
                }
                Ok(unset)
            }
        }

        impl bitflags::Bits for FeatureBits {
            const EMPTY: Self = $name::empty().bits();

            const ALL: Self = $name::all().bits();
        }

        impl Flags for $name {
            const FLAGS: &'static [bitflags::Flag<Self>] = $name::FLAGS;

            type Bits = FeatureBits;

            fn bits(&self) -> FeatureBits {
                FeatureBits([
                    $(self.$lower_inner_name.bits(),)*
                ])
            }

            fn from_bits_retain(bits: FeatureBits) -> Self {
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
            pub const fn bits(&self) -> FeatureBits {
                FeatureBits([
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

            bitflags_independent_two_arg! {
                /// Bitwise or - `self | other`
                union $($lower_inner_name)*
            }

            bitflags_independent_two_arg! {
                /// Bitwise and - `self & other`
                intersection $($lower_inner_name)*
            }

            bitflags_independent_two_arg! {
                /// Bitwise and of the complement of other - `self & !other`
                difference $($lower_inner_name)*
            }

            bitflags_independent_two_arg! {
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
            pub const fn from_bits(bits:FeatureBits) -> Option<Self> {
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
            pub const fn from_bits_truncate(bits:FeatureBits) -> Self {
                let [$($lower_inner_name,)*] = bits.0;
                Self { $($lower_inner_name: $inner_name::from_bits_truncate($lower_inner_name),)* }
            }

            /// Takes in [`FeatureBits`] and returns Self with all bits that were set without removing
            /// invalid bits
            pub const fn from_bits_retain(bits:FeatureBits) -> Self {
                let [$($lower_inner_name,)*] = bits.0;
                Self { $($lower_inner_name: $inner_name::from_bits_retain($lower_inner_name),)* }
            }

            /// Takes in a name and returns Self if it matches or none if the name does not match
            /// the name of any of the flags. Name is capitalisation dependent.
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
            pub const fn iter_names(&self) -> bitflags::iter::IterNames<$name> {
                bitflags::iter::IterNames::__private_const_new($name::FLAGS, *self, *self)
            }

            $(
                $(
                    $(#[$inner $($args)*])*
                    // We need this for structs with only a member.
                    #[allow(clippy::needless_update)]
                    pub const $Flag: Self = Self {
                        $lower_inner_name: $inner_name::from_bits_truncate($value),
                        ..Self::empty()
                    };
                )*
            )*
        }

        $(
            impl From<$inner_name> for Features {
                // We need this for structs with only a member.
                #[allow(clippy::needless_update)]
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

impl From<FeatureBits> for Features {
    fn from(value: FeatureBits) -> Self {
        Self::from_bits_retain(value)
    }
}

impl From<Features> for FeatureBits {
    fn from(value: Features) -> Self {
        value.bits()
    }
}

bitflags_array! {
    /// Features that are not guaranteed to be supported.
    ///
    /// These are either part of the webgpu standard, or are extension features supported by
    /// wgpu when targeting native.
    ///
    /// If you want to use a feature, you need to first verify that the adapter supports
    /// the feature. If the adapter does not support the feature, requesting a device with it enabled
    /// will panic.
    ///
    /// Corresponds to [WebGPU `GPUFeatureName`](
    /// https://gpuweb.github.io/gpuweb/#enumdef-gpufeaturename).
    #[repr(C)]
    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct Features: [u64; 2];

    /// Features that are not guaranteed to be supported.
    ///
    /// These are extension features supported by wgpu when targeting native. For all features see [`Features`]
    ///
    /// If you want to use a feature, you need to first verify that the adapter supports
    /// the feature. If the adapter does not support the feature, requesting a device with it enabled
    /// will panic.
    ///
    /// Corresponds to [WebGPU `GPUFeatureName`](
    /// https://gpuweb.github.io/gpuweb/#enumdef-gpufeaturename).
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FeaturesWGPU features_wgpu {
        /// Allows shaders to use f32 atomic load, store, add, sub, and exchange.
        ///
        /// Supported platforms:
        /// - Metal (with MSL 3.0+ and Apple7+/Mac2)
        /// - Vulkan (with [VK_EXT_shader_atomic_float])
        ///
        /// This is a native only feature.
        ///
        /// [VK_EXT_shader_atomic_float]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_atomic_float.html
        const SHADER_FLOAT32_ATOMIC = 1 << 0;

        // The features starting with a ? are features that might become part of the spec or
        // at the very least we can implement as native features; since they should cover all
        // possible formats and capabilities across backends.
        //
        // ? const FORMATS_TIER_1 = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3837)
        // ? const RW_STORAGE_TEXTURE_TIER_1 = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3838)
        // ? const NORM16_FILTERABLE = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3839)
        // ? const NORM16_RESOLVE = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3839)
        // ? const FLOAT32_BLENDABLE = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3556)
        // ? const 32BIT_FORMAT_MULTISAMPLE = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3844)
        // ? const 32BIT_FORMAT_RESOLVE = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3844)
        // ? const TEXTURE_COMPRESSION_ASTC_HDR = 1 << ??; (https://github.com/gpuweb/gpuweb/issues/3856)
        // TEXTURE_FORMAT_16BIT_NORM & TEXTURE_COMPRESSION_ASTC_HDR will most likely become web features as well
        // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES might not be necessary if we have all the texture features implemented

        // Texture Formats:

        /// Enables normalized `16-bit` texture formats.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const TEXTURE_FORMAT_16BIT_NORM = 1 << 1;
        /// Enables ASTC HDR family of compressed textures.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for ASTC formats with the HDR channel type.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// Supported Platforms:
        /// - Metal
        /// - Vulkan
        /// - OpenGL
        ///
        /// This is a native only feature.
        const TEXTURE_COMPRESSION_ASTC_HDR = 1 << 2;
        /// Enables device specific texture format features.
        ///
        /// See `TextureFormatFeatures` for a listing of the features in question.
        ///
        /// By default only texture format properties as defined by the WebGPU specification are allowed.
        /// Enabling this feature flag extends the features of each format to the ones supported by the current device.
        /// Note that without this flag, read/write storage access is not allowed at all.
        ///
        /// This extension does not enable additional formats.
        ///
        /// This is a native only feature.
        const TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES = 1 << 3;

        // API:

        /// Enables use of Pipeline Statistics Queries. These queries tell the count of various operations
        /// performed between the start and stop call. Call [`RenderPass::begin_pipeline_statistics_query`] to start
        /// a query, then call [`RenderPass::end_pipeline_statistics_query`] to stop one.
        ///
        /// They must be resolved using [`CommandEncoder::resolve_query_set`] into a buffer.
        /// The rules on how these resolve into buffers are detailed in the documentation for [`PipelineStatisticsTypes`].
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - DX12
        ///
        /// This is a native only feature with a [proposal](https://github.com/gpuweb/gpuweb/blob/0008bd30da2366af88180b511a5d0d0c1dffbc36/proposals/pipeline-statistics-query.md) for the web.
        ///
        /// [`RenderPass::begin_pipeline_statistics_query`]: https://docs.rs/wgpu/latest/wgpu/struct.RenderPass.html#method.begin_pipeline_statistics_query
        /// [`RenderPass::end_pipeline_statistics_query`]: https://docs.rs/wgpu/latest/wgpu/struct.RenderPass.html#method.end_pipeline_statistics_query
        /// [`CommandEncoder::resolve_query_set`]: https://docs.rs/wgpu/latest/wgpu/struct.CommandEncoder.html#method.resolve_query_set
        /// [`PipelineStatisticsTypes`]: super::PipelineStatisticsTypes
        const PIPELINE_STATISTICS_QUERY = 1 << 4;
        /// Allows for timestamp queries directly on command encoders.
        ///
        /// Implies [`Features::TIMESTAMP_QUERY`] is supported.
        ///
        /// Additionally allows for timestamp writes on command encoders
        /// using  [`CommandEncoder::write_timestamp`].
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        ///
        /// [`CommandEncoder::write_timestamp`]: https://docs.rs/wgpu/latest/wgpu/struct.CommandEncoder.html#method.write_timestamp
        const TIMESTAMP_QUERY_INSIDE_ENCODERS = 1 << 5;
        /// Allows for timestamp queries directly on command encoders.
        ///
        /// Implies [`Features::TIMESTAMP_QUERY`] & [`Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] is supported.
        ///
        /// Additionally allows for timestamp queries to be used inside render & compute passes using:
        /// - [`RenderPass::write_timestamp`]
        /// - [`ComputePass::write_timestamp`]
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal (AMD & Intel, not Apple GPUs)
        ///
        /// This is generally not available on tile-based rasterization GPUs.
        ///
        /// This is a native only feature with a [proposal](https://github.com/gpuweb/gpuweb/blob/0008bd30da2366af88180b511a5d0d0c1dffbc36/proposals/timestamp-query-inside-passes.md) for the web.
        ///
        /// [`RenderPass::write_timestamp`]: https://docs.rs/wgpu/latest/wgpu/struct.RenderPass.html#method.write_timestamp
        /// [`ComputePass::write_timestamp`]: https://docs.rs/wgpu/latest/wgpu/struct.ComputePass.html#method.write_timestamp
        const TIMESTAMP_QUERY_INSIDE_PASSES = 1 << 6;
        /// Webgpu only allows the MAP_READ and MAP_WRITE buffer usage to be matched with
        /// COPY_DST and COPY_SRC respectively. This removes this requirement.
        ///
        /// This is only beneficial on systems that share memory between CPU and GPU. If enabled
        /// on a system that doesn't, this can severely hinder performance. Only use if you understand
        /// the consequences.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const MAPPABLE_PRIMARY_BUFFERS = 1 << 7;
        /// Allows the user to create uniform arrays of textures in shaders:
        ///
        /// ex.
        /// - `var textures: binding_array<texture_2d<f32>, 10>` (WGSL)
        /// - `uniform texture2D textures[10]` (GLSL)
        ///
        /// If [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] is supported as well as this, the user
        /// may also create uniform arrays of storage textures.
        ///
        /// ex.
        /// - `var textures: array<texture_storage_2d<r32float, write>, 10>` (WGSL)
        /// - `uniform image2D textures[10]` (GLSL)
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan
        ///
        /// This is a native only feature.
        const TEXTURE_BINDING_ARRAY = 1 << 8;
        /// Allows the user to create arrays of buffers in shaders:
        ///
        /// ex.
        /// - `var<uniform> buffer_array: array<MyBuffer, 10>` (WGSL)
        /// - `uniform myBuffer { ... } buffer_array[10]` (GLSL)
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// If [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] is supported as well as this, the user
        /// may also create arrays of storage buffers.
        ///
        /// ex.
        /// - `var<storage> buffer_array: array<MyBuffer, 10>` (WGSL)
        /// - `buffer myBuffer { ... } buffer_array[10]` (GLSL)
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const BUFFER_BINDING_ARRAY = 1 << 9;
        /// Allows the user to create uniform arrays of storage buffers or textures in shaders,
        /// if resp. [`Features::BUFFER_BINDING_ARRAY`] or [`Features::TEXTURE_BINDING_ARRAY`]
        /// is supported.
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// Supported platforms:
        /// - Metal (with MSL 2.2+ on macOS 10.13+)
        /// - Vulkan
        ///
        /// This is a native only feature.
        const STORAGE_RESOURCE_BINDING_ARRAY = 1 << 10;
        /// Allows shaders to index sampled texture and storage buffer resource arrays with dynamically non-uniform values:
        ///
        /// ex. `texture_array[vertex_data]`
        ///
        /// In order to use this capability, the corresponding GLSL extension must be enabled like so:
        ///
        /// `#extension GL_EXT_nonuniform_qualifier : require`
        ///
        /// and then used either as `nonuniformEXT` qualifier in variable declaration:
        ///
        /// ex. `layout(location = 0) nonuniformEXT flat in int vertex_data;`
        ///
        /// or as `nonuniformEXT` constructor:
        ///
        /// ex. `texture_array[nonuniformEXT(vertex_data)]`
        ///
        /// WGSL and HLSL do not need any extension.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s shaderSampledImageArrayNonUniformIndexing & shaderStorageBufferArrayNonUniformIndexing feature)
        ///
        /// This is a native only feature.
        const SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING = 1 << 11;
        /// Allows shaders to index storage texture resource arrays with dynamically non-uniform values:
        ///
        /// ex. `texture_array[vertex_data]`
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s shaderStorageTextureArrayNonUniformIndexing feature)
        ///
        /// This is a native only feature.
        const STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING = 1 << 12;
        /// Allows the user to create bind groups containing arrays with less bindings than the BindGroupLayout.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        ///
        /// This is a native only feature.
        const PARTIALLY_BOUND_BINDING_ARRAY = 1 << 13;
        /// Allows the user to call [`RenderPass::multi_draw_indirect`] and [`RenderPass::multi_draw_indexed_indirect`].
        ///
        /// Allows multiple indirect calls to be dispatched from a single buffer.
        ///
        /// Natively Supported Platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// Emulated Platforms:
        /// - Metal
        /// - OpenGL
        /// - WebGPU
        ///
        /// Emulation is preformed by looping over the individual indirect draw calls in the backend. This is still significantly
        /// faster than enulating it yourself, as wgpu only does draw call validation once.
        ///
        /// [`RenderPass::multi_draw_indirect`]: ../wgpu/struct.RenderPass.html#method.multi_draw_indirect
        /// [`RenderPass::multi_draw_indexed_indirect`]: ../wgpu/struct.RenderPass.html#method.multi_draw_indexed_indirect
        const MULTI_DRAW_INDIRECT = 1 << 14;
        /// Allows the user to call [`RenderPass::multi_draw_indirect_count`] and [`RenderPass::multi_draw_indexed_indirect_count`].
        ///
        /// This allows the use of a buffer containing the actual number of draw calls.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan 1.2+ (or VK_KHR_draw_indirect_count)
        ///
        /// This is a native only feature.
        ///
        /// [`RenderPass::multi_draw_indirect_count`]: ../wgpu/struct.RenderPass.html#method.multi_draw_indirect_count
        /// [`RenderPass::multi_draw_indexed_indirect_count`]: ../wgpu/struct.RenderPass.html#method.multi_draw_indexed_indirect_count
        const MULTI_DRAW_INDIRECT_COUNT = 1 << 15;
        /// Allows the use of push constants: small, fast bits of memory that can be updated
        /// inside a [`RenderPass`].
        ///
        /// Allows the user to call [`RenderPass::set_push_constants`], provide a non-empty array
        /// to [`PipelineLayoutDescriptor`], and provide a non-zero limit to [`Limits::max_push_constant_size`].
        ///
        /// A block of push constants can be declared in WGSL with `var<push_constant>`:
        ///
        /// ```rust,ignore
        /// struct PushConstants { example: f32, }
        /// var<push_constant> c: PushConstants;
        /// ```
        ///
        /// In GLSL, this corresponds to `layout(push_constant) uniform Name {..}`.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal
        /// - OpenGL (emulated with uniforms)
        ///
        /// This is a native only feature.
        ///
        /// [`RenderPass`]: ../wgpu/struct.RenderPass.html
        /// [`PipelineLayoutDescriptor`]: ../wgpu/struct.PipelineLayoutDescriptor.html
        /// [`RenderPass::set_push_constants`]: ../wgpu/struct.RenderPass.html#method.set_push_constants
        /// [`Limits::max_push_constant_size`]: super::Limits
        const PUSH_CONSTANTS = 1 << 16;
        /// Allows the use of [`AddressMode::ClampToBorder`] with a border color
        /// of [`SamplerBorderColor::Zero`].
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal
        /// - OpenGL
        ///
        /// This is a native only feature.
        ///
        /// [`AddressMode::ClampToBorder`]: super::AddressMode::ClampToBorder
        /// [`SamplerBorderColor::Zero`]: super::SamplerBorderColor::Zero
        const ADDRESS_MODE_CLAMP_TO_ZERO = 1 << 17;
        /// Allows the use of [`AddressMode::ClampToBorder`] with a border color
        /// other than [`SamplerBorderColor::Zero`].
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal (macOS 10.12+ only)
        /// - OpenGL
        ///
        /// This is a native only feature.
        ///
        /// [`AddressMode::ClampToBorder`]: super::AddressMode::ClampToBorder
        /// [`SamplerBorderColor::Zero`]: super::SamplerBorderColor::Zero
        const ADDRESS_MODE_CLAMP_TO_BORDER = 1 << 18;
        /// Allows the user to set [`PolygonMode::Line`] in [`PrimitiveState::polygon_mode`]
        ///
        /// This allows drawing polygons/triangles as lines (wireframe) instead of filled
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal
        ///
        /// This is a native only feature.
        ///
        /// [`PrimitiveState::polygon_mode`]: super::PrimitiveState
        /// [`PolygonMode::Line`]: super::PolygonMode::Line
        const POLYGON_MODE_LINE = 1 << 19;
        /// Allows the user to set [`PolygonMode::Point`] in [`PrimitiveState::polygon_mode`]
        ///
        /// This allows only drawing the vertices of polygons/triangles instead of filled
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        ///
        /// [`PrimitiveState::polygon_mode`]: super::PrimitiveState
        /// [`PolygonMode::Point`]: super::PolygonMode::Point
        const POLYGON_MODE_POINT = 1 << 20;
        /// Allows the user to set a overestimation-conservative-rasterization in [`PrimitiveState::conservative`]
        ///
        /// Processing of degenerate triangles/lines is hardware specific.
        /// Only triangles are supported.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        ///
        /// [`PrimitiveState::conservative`]: super::PrimitiveState::conservative
        const CONSERVATIVE_RASTERIZATION = 1 << 21;
        /// Enables bindings of writable storage buffers and textures visible to vertex shaders.
        ///
        /// Note: some (tiled-based) platforms do not support vertex shaders with any side-effects.
        ///
        /// Supported Platforms:
        /// - All
        ///
        /// This is a native only feature.
        const VERTEX_WRITABLE_STORAGE = 1 << 22;
        /// Enables clear to zero for textures.
        ///
        /// Supported platforms:
        /// - All
        ///
        /// This is a native only feature.
        const CLEAR_TEXTURE = 1 << 23;
        /// Enables creating shader modules from Metal MSL computer shaders (unsafe).
        ///
        /// Metal data is not parsed or interpreted in any way
        ///
        /// Supported platforms:
        /// - Metal
        ///
        /// This is a native only feature.
        const MSL_SHADER_PASSTHROUGH = 1 << 24;
        /// Enables creating shader modules from SPIR-V binary data (unsafe).
        ///
        /// SPIR-V data is not parsed or interpreted in any way; you can use
        /// [`wgpu::make_spirv_raw!`] to check for alignment and magic number when converting from
        /// raw bytes.
        ///
        /// Supported platforms:
        /// - Vulkan, in case shader's requested capabilities and extensions agree with
        /// Vulkan implementation.
        ///
        /// This is a native only feature.
        ///
        /// [`wgpu::make_spirv_raw!`]: https://docs.rs/wgpu/latest/wgpu/macro.include_spirv_raw.html
        const SPIRV_SHADER_PASSTHROUGH = 1 << 25;
        /// Enables multiview render passes and `builtin(view_index)` in vertex shaders.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - OpenGL (web only)
        ///
        /// This is a native only feature.
        const MULTIVIEW = 1 << 26;
        /// Enables using 64-bit types for vertex attributes.
        ///
        /// Requires SHADER_FLOAT64.
        ///
        /// Supported Platforms: N/A
        ///
        /// This is a native only feature.
        const VERTEX_ATTRIBUTE_64BIT = 1 << 27;
        /// Enables image atomic fetch add, and, xor, or, min, and max for R32Uint and R32Sint textures.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal (with MSL 3.1+)
        ///
        /// This is a native only feature.
        const TEXTURE_ATOMIC = 1 << 28;
        /// Allows for creation of textures of format [`TextureFormat::NV12`]
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        ///
        /// [`TextureFormat::NV12`]: super::TextureFormat::NV12
        const TEXTURE_FORMAT_NV12 = 1 << 29;
        /// Allows for creation of textures of format [`TextureFormat::P010`]
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        ///
        /// [`TextureFormat::P010`]: super::TextureFormat::P010
        const TEXTURE_FORMAT_P010 = 1 << 30;

        /// Allows for the creation and usage of `ExternalTexture`s, and bind
        /// group layouts containing external texture `BindingType`s.
        ///
        /// Conceptually this should really be a [`crate::DownlevelFlags`] as
        /// it corresponds to WebGPU's [`GPUExternalTexture`](
        /// https://www.w3.org/TR/webgpu/#gpuexternaltexture).
        /// However, the implementation is currently in-progress, and until it
        /// is complete we do not want applications to ignore adapters due to
        /// a missing downlevel flag, when they may not require this feature at
        /// all.
        ///
        /// Supported platforms:
        /// - DX12
        const EXTERNAL_TEXTURE = 1 << 31;

        // Shader:

        /// ***THIS IS EXPERIMENTAL:*** Features enabled by this may have
        /// major bugs in it and are expected to be subject to breaking changes, suggestions
        /// for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/1040)
        ///
        /// Allows for the creation of ray-tracing queries within shaders.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native-only feature.
        const EXPERIMENTAL_RAY_QUERY = 1 << 32;
        /// Enables 64-bit floating point types in SPIR-V shaders.
        ///
        /// Note: even when supported by GPU hardware, 64-bit floating point operations are
        /// frequently between 16 and 64 _times_ slower than equivalent operations on 32-bit floats.
        ///
        /// Supported Platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const SHADER_F64 = 1 << 33;
        /// Allows shaders to use i16. Not currently supported in `naga`, only available through `spirv-passthrough`.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const SHADER_I16 = 1 << 34;
        /// Enables `builtin(primitive_index)` in fragment shaders.
        ///
        /// Note: enables geometry processing for pipelines using the builtin.
        /// This may come with a significant performance impact on some hardware.
        /// Other pipelines are not affected.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal (some)
        /// - OpenGL (some)
        ///
        /// This is a native only feature.
        const SHADER_PRIMITIVE_INDEX = 1 << 35;
        /// Allows shaders to use the `early_depth_test` attribute.
        ///
        /// The attribute is applied to the fragment shader entry point. It can be used in two
        /// ways:
        ///
        ///   1. Force early depth/stencil tests:
        ///
        ///      - `@early_depth_test(force)` (WGSL)
        ///
        ///      - `layout(early_fragment_tests) in;` (GLSL)
        ///
        ///   2. Provide a conservative depth specifier that allows an additional early
        ///      depth test under certain conditions:
        ///
        ///      - `@early_depth_test(greater_equal/less_equal/unchanged)` (WGSL)
        ///
        ///      - `layout(depth_<greater/less/unchanged>) out float gl_FragDepth;` (GLSL)
        ///
        /// See [`EarlyDepthTest`] for more details.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - GLES 3.1+
        ///
        /// This is a native only feature.
        ///
        /// [`EarlyDepthTest`]: https://docs.rs/naga/latest/naga/ir/enum.EarlyDepthTest.html
        const SHADER_EARLY_DEPTH_TEST = 1 << 36;
        /// Allows shaders to use i64 and u64.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12 (DXC only)
        /// - Metal (with MSL 2.3+)
        ///
        /// This is a native only feature.
        const SHADER_INT64 = 1 << 37;
        /// Allows compute and fragment shaders to use the subgroup operation built-ins
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const SUBGROUP = 1 << 38;
        /// Allows vertex shaders to use the subgroup operation built-ins
        ///
        /// Supported Platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const SUBGROUP_VERTEX = 1 << 39;
        /// Allows shaders to use the subgroup barrier
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - Metal
        ///
        /// This is a native only feature.
        const SUBGROUP_BARRIER = 1 << 40;
        /// Allows the use of pipeline cache objects
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// Unimplemented Platforms:
        /// - DX12
        /// - Metal
        const PIPELINE_CACHE = 1 << 41;
        /// Allows shaders to use i64 and u64 atomic min and max.
        ///
        /// Supported platforms:
        /// - Vulkan (with VK_KHR_shader_atomic_int64)
        /// - DX12 (with SM 6.6+)
        /// - Metal (with MSL 2.4+)
        ///
        /// This is a native only feature.
        const SHADER_INT64_ATOMIC_MIN_MAX = 1 << 42;
        /// Allows shaders to use all i64 and u64 atomic operations.
        ///
        /// Supported platforms:
        /// - Vulkan (with VK_KHR_shader_atomic_int64)
        /// - DX12 (with SM 6.6+)
        ///
        /// This is a native only feature.
        const SHADER_INT64_ATOMIC_ALL_OPS = 1 << 43;
        /// Allows using the [VK_GOOGLE_display_timing] Vulkan extension.
        ///
        /// This is used for frame pacing to reduce latency, and is generally only available on Android.
        ///
        /// This feature does not have a `wgpu`-level API, and so users of wgpu wishing
        /// to use this functionality must access it using various `as_hal` functions,
        /// primarily [`Surface::as_hal()`], to then use.
        ///
        /// Supported platforms:
        /// - Vulkan (with [VK_GOOGLE_display_timing])
        ///
        /// This is a native only feature.
        ///
        /// [VK_GOOGLE_display_timing]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_display_timing.html
        /// [`Surface::as_hal()`]: https://docs.rs/wgpu/latest/wgpu/struct.Surface.html#method.as_hal
        const VULKAN_GOOGLE_DISPLAY_TIMING = 1 << 44;

        /// Allows using the [VK_KHR_external_memory_win32] Vulkan extension.
        ///
        /// Supported platforms:
        /// - Vulkan (with [VK_KHR_external_memory_win32])
        ///
        /// This is a native only feature.
        ///
        /// [VK_KHR_external_memory_win32]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_external_memory_win32.html
        const VULKAN_EXTERNAL_MEMORY_WIN32 = 1 << 45;

        /// Enables R64Uint image atomic min and max.
        ///
        /// Supported platforms:
        /// - Vulkan (with VK_EXT_shader_image_atomic_int64)
        /// - DX12 (with SM 6.6+)
        /// - Metal (with MSL 3.1+)
        ///
        /// This is a native only feature.
        const TEXTURE_INT64_ATOMIC = 1 << 46;

        /// Allows uniform buffers to be bound as binding arrays.
        ///
        /// This allows:
        /// - Shaders to contain `var<uniform> buffer: binding_array<UniformBuffer>;`
        /// - The `count` field of `BindGroupLayoutEntry`s with `Uniform` buffers, to be set to `Some`.
        ///
        /// Supported platforms:
        /// - None (<https://github.com/gfx-rs/wgpu/issues/7149>)
        ///
        /// Potential Platforms:
        /// - DX12
        /// - Metal
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s `shaderUniformBufferArrayNonUniformIndexing` feature)
        ///
        /// This is a native only feature.
        const UNIFORM_BUFFER_BINDING_ARRAYS = 1 << 47;

        /// Enables mesh shaders and task shaders in mesh shader pipelines.
        ///
        /// Supported platforms:
        /// - Vulkan (with [VK_EXT_mesh_shader](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html))
        ///
        /// Potential Platforms:
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const EXPERIMENTAL_MESH_SHADER = 1 << 48;

        /// ***THIS IS EXPERIMENTAL:*** Features enabled by this may have
        /// major bugs in them and are expected to be subject to breaking changes, suggestions
        /// for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/6762)
        ///
        /// Allows for returning of the hit triangle's vertex position when tracing with an
        /// acceleration structure marked with [`AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN`].
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature
        ///
        /// [`AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN`]: super::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN
        const EXPERIMENTAL_RAY_HIT_VERTEX_RETURN = 1 << 49;

        /// Enables multiview in mesh shader pipelines
        ///
        /// Supported platforms:
        /// - Vulkan (with [VK_EXT_mesh_shader](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_mesh_shader.html))
        ///
        /// Potential Platforms:
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const EXPERIMENTAL_MESH_SHADER_MULTIVIEW = 1 << 50;

        /// Allows usage of additional vertex formats in [BlasTriangleGeometrySizeDescriptor::vertex_format]
        ///
        /// Supported platforms
        /// - Vulkan
        /// - DX12
        ///
        /// [BlasTriangleGeometrySizeDescriptor::vertex_format]: super::BlasTriangleGeometrySizeDescriptor
        const EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS = 1 << 51;

        /// Enables creating shader modules from DirectX HLSL or DXIL shaders (unsafe)
        ///
        /// HLSL/DXIL data is not parsed or interpreted in any way
        ///
        /// Supported platforms:
        /// - DX12
        ///
        /// This is a native only feature.
        const HLSL_DXIL_SHADER_PASSTHROUGH = 1 << 52;
    }

    /// Features that are not guaranteed to be supported.
    ///
    /// These are part of the webgpu standard. For all features see [`Features`]
    ///
    /// If you want to use a feature, you need to first verify that the adapter supports
    /// the feature. If the adapter does not support the feature, requesting a device with it enabled
    /// will panic.
    ///
    /// Corresponds to [WebGPU `GPUFeatureName`](
    /// https://gpuweb.github.io/gpuweb/#enumdef-gpufeaturename).
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FeaturesWebGPU features_webgpu {
        // API:

        /// By default, polygon depth is clipped to 0-1 range before/during rasterization.
        /// Anything outside of that range is rejected, and respective fragments are not touched.
        ///
        /// With this extension, we can disabling clipping. That allows
        /// shadow map occluders to be rendered into a tighter depth range.
        ///
        /// Supported platforms:
        /// - desktops
        /// - some mobile chips
        ///
        /// This is a web and native feature.
        const DEPTH_CLIP_CONTROL = WEBGPU_FEATURE_DEPTH_CLIP_CONTROL;

        /// Allows for explicit creation of textures of format [`TextureFormat::Depth32FloatStencil8`]
        ///
        /// Supported platforms:
        /// - Vulkan (mostly)
        /// - DX12
        /// - Metal
        /// - OpenGL
        ///
        /// This is a web and native feature.
        ///
        /// [`TextureFormat::Depth32FloatStencil8`]: super::TextureFormat::Depth32FloatStencil8
        const DEPTH32FLOAT_STENCIL8 = WEBGPU_FEATURE_DEPTH32FLOAT_STENCIL8;

        /// Enables BCn family of compressed textures. All BCn textures use 4x4 pixel blocks
        /// with 8 or 16 bytes per block.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for BCn formats.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// This feature guarantees availability of sliced-3d textures for BC formats when combined with TEXTURE_COMPRESSION_BC_SLICED_3D.
        ///
        /// Supported Platforms:
        /// - desktops
        /// - Mobile (All Apple9 and some Apple7 and Apple8 devices)
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_BC = WEBGPU_FEATURE_TEXTURE_COMPRESSION_BC;


        /// Allows the 3d dimension for textures with BC compressed formats.
        ///
        /// This feature must be used in combination with TEXTURE_COMPRESSION_BC to enable 3D textures with BC compression.
        /// It does not enable the BC formats by itself.
        ///
        /// Supported Platforms:
        /// - desktops
        /// - Mobile (All Apple9 and some Apple7 and Apple8 devices)
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_BC_SLICED_3D = WEBGPU_FEATURE_TEXTURE_COMPRESSION_BC_SLICED_3D;

        /// Enables ETC family of compressed textures. All ETC textures use 4x4 pixel blocks.
        /// ETC2 RGB and RGBA1 are 8 bytes per block. RTC2 RGBA8 and EAC are 16 bytes per block.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for ETC2 formats.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// Supported Platforms:
        /// - Vulkan on Intel
        /// - Mobile (some)
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_ETC2 = WEBGPU_FEATURE_TEXTURE_COMPRESSION_ETC2;

        /// Enables ASTC family of compressed textures. ASTC textures use pixel blocks varying from 4x4 to 12x12.
        /// Blocks are always 16 bytes.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for ASTC formats with Unorm/UnormSrgb channel type.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// This feature does not guarantee availability of sliced 3d textures for ASTC formats.
        /// If available, 3d support can be enabled by TEXTURE_COMPRESSION_ASTC_SLICED_3D feature.
        ///
        /// Supported Platforms:
        /// - Vulkan on Intel
        /// - Mobile (some)
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_ASTC = WEBGPU_FEATURE_TEXTURE_COMPRESSION_ASTC;


        /// Allows the 3d dimension for textures with ASTC compressed formats.
        ///
        /// This feature must be used in combination with TEXTURE_COMPRESSION_ASTC to enable 3D textures with ASTC compression.
        /// It does not enable the ASTC formats by itself.
        ///
        /// Supported Platforms:
        /// - Vulkan (some)
        /// - Metal on Apple3+
        /// - OpenGL/WebGL (some)
        ///
        /// Not Supported:
        /// - DX12
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_ASTC_SLICED_3D = WEBGPU_FEATURE_TEXTURE_COMPRESSION_ASTC_SLICED_3D;

        /// Enables use of Timestamp Queries. These queries tell the current gpu timestamp when
        /// all work before the query is finished.
        ///
        /// This feature allows the use of
        /// - [`RenderPassDescriptor::timestamp_writes`]
        /// - [`ComputePassDescriptor::timestamp_writes`]
        /// to write out timestamps.
        ///
        /// For arbitrary timestamp write commands on encoders refer to [`Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`].
        /// For arbitrary timestamp write commands on passes refer to [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`].
        ///
        /// They must be resolved using [`CommandEncoder::resolve_query_set`] into a buffer,
        /// then the result must be multiplied by the timestamp period [`Queue::get_timestamp_period`]
        /// to get the timestamp in nanoseconds. Multiple timestamps can then be diffed to get the
        /// time for operations between them to finish.
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a web and native feature.
        ///
        /// [`RenderPassDescriptor::timestamp_writes`]: https://docs.rs/wgpu/latest/wgpu/struct.RenderPassDescriptor.html#structfield.timestamp_writes
        /// [`ComputePassDescriptor::timestamp_writes`]: https://docs.rs/wgpu/latest/wgpu/struct.ComputePassDescriptor.html#structfield.timestamp_writes
        /// [`CommandEncoder::resolve_query_set`]: https://docs.rs/wgpu/latest/wgpu/struct.CommandEncoder.html#method.resolve_query_set
        /// [`Queue::get_timestamp_period`]: https://docs.rs/wgpu/latest/wgpu/struct.Queue.html#method.get_timestamp_period
        const TIMESTAMP_QUERY = WEBGPU_FEATURE_TIMESTAMP_QUERY;

        /// Allows non-zero value for the `first_instance` member in indirect draw calls.
        ///
        /// If this feature is not enabled, and the `first_instance` member is non-zero, the behavior may be:
        /// - The draw call is ignored.
        /// - The draw call is executed as if the `first_instance` is zero.
        /// - The draw call is executed with the correct `first_instance` value.
        ///
        /// Supported Platforms:
        /// - Vulkan (mostly)
        /// - DX12
        /// - Metal on Apple3+ or Mac1+
        /// - OpenGL (Desktop 4.2+ with ARB_shader_draw_parameters only)
        ///
        /// Not Supported:
        /// - OpenGL ES / WebGL
        ///
        /// This is a web and native feature.
        const INDIRECT_FIRST_INSTANCE = WEBGPU_FEATURE_INDIRECT_FIRST_INSTANCE;

        /// Allows shaders to use 16-bit floating point types. You may use them uniform buffers,
        /// storage buffers, and local variables. You may not use them in push constants.
        ///
        /// In order to use this in WGSL shaders, you must add `enable f16;` to the top of your shader,
        /// before any global items.
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - Metal
        /// - DX12
        ///
        /// This is a web and native feature.
        const SHADER_F16 = WEBGPU_FEATURE_SHADER_F16;

        /// Allows for usage of textures of format [`TextureFormat::Rg11b10Ufloat`] as a render target
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a web and native feature.
        ///
        /// [`TextureFormat::Rg11b10Ufloat`]: super::TextureFormat::Rg11b10Ufloat
        const RG11B10UFLOAT_RENDERABLE = WEBGPU_FEATURE_RG11B10UFLOAT_RENDERABLE;

        /// Allows the [`TextureUsages::STORAGE_BINDING`] usage on textures with format [`TextureFormat::Bgra8Unorm`]
        ///
        /// Supported Platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a web and native feature.
        ///
        /// [`TextureFormat::Bgra8Unorm`]: super::TextureFormat::Bgra8Unorm
        /// [`TextureUsages::STORAGE_BINDING`]: super::TextureUsages::STORAGE_BINDING
        const BGRA8UNORM_STORAGE = WEBGPU_FEATURE_BGRA8UNORM_STORAGE;


        /// Allows textures with formats "r32float", "rg32float", and "rgba32float" to be filterable.
        ///
        /// Supported Platforms:
        /// - Vulkan (mainly on Desktop GPUs)
        /// - DX12
        /// - Metal on macOS or Apple9+ GPUs, optional on iOS/iPadOS with Apple7/8 GPUs
        /// - GL with one of `GL_ARB_color_buffer_float`/`GL_EXT_color_buffer_float`/`OES_texture_float_linear`
        ///
        /// This is a web and native feature.
        const FLOAT32_FILTERABLE = WEBGPU_FEATURE_FLOAT32_FILTERABLE;

        /// Allows two outputs from a shader to be used for blending.
        /// Note that dual-source blending doesn't support multiple render targets.
        ///
        /// For more info see the OpenGL ES extension GL_EXT_blend_func_extended.
        ///
        /// Supported platforms:
        /// - OpenGL ES (with GL_EXT_blend_func_extended)
        /// - Metal (with MSL 1.2+)
        /// - Vulkan (with dualSrcBlend)
        /// - DX12
        ///
        /// This is a web and native feature.
        const DUAL_SOURCE_BLENDING = WEBGPU_FEATURE_DUAL_SOURCE_BLENDING;

        /// Allows the use of `@builtin(clip_distances)` in WGSL.
        ///
        /// Supported platforms:
        /// - Vulkan (mainly on Desktop GPUs)
        /// - GL (Desktop or `GL_EXT_clip_cull_distance`)
        ///
        /// This is a web and native feature.
        const CLIP_DISTANCES = WEBGPU_FEATURE_CLIP_DISTANCES;
    }
}

impl Features {
    /// Mask of all features which are part of the upstream WebGPU standard.
    #[must_use]
    pub const fn all_webgpu_mask() -> Self {
        Self::from_bits_truncate(FeatureBits([
            FeaturesWGPU::empty().bits(),
            FeaturesWebGPU::all().bits(),
        ]))
    }

    /// Mask of all features that are only available when targeting native (not web).
    #[must_use]
    pub const fn all_native_mask() -> Self {
        Self::from_bits_truncate(FeatureBits([
            FeaturesWGPU::all().bits(),
            FeaturesWebGPU::empty().bits(),
        ]))
    }

    /// Vertex formats allowed for creating and building BLASes
    #[must_use]
    pub fn allowed_vertex_formats_for_blas(&self) -> Vec<VertexFormat> {
        let mut formats = Vec::new();
        if self.intersects(Self::EXPERIMENTAL_RAY_QUERY) {
            formats.push(VertexFormat::Float32x3);
        }
        if self.contains(Self::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS) {
            formats.push(VertexFormat::Float32x2);
            formats.push(VertexFormat::Float16x2);
            formats.push(VertexFormat::Float16x4);
            formats.push(VertexFormat::Snorm16x2);
            formats.push(VertexFormat::Snorm16x4);
        }
        formats
    }
}

#[cfg(test)]
mod tests {
    use crate::{Features, FeaturesWGPU, FeaturesWebGPU};

    #[cfg(feature = "serde")]
    #[test]
    fn check_hex() {
        use crate::FeatureBits;

        use bitflags::{
            parser::{ParseHex as _, WriteHex as _},
            Bits as _,
        };

        let mut hex = alloc::string::String::new();
        FeatureBits::ALL.write_hex(&mut hex).unwrap();
        assert_eq!(
            FeatureBits::parse_hex(hex.as_str()).unwrap(),
            FeatureBits::ALL
        );

        hex.clear();
        FeatureBits::EMPTY.write_hex(&mut hex).unwrap();
        assert_eq!(
            FeatureBits::parse_hex(hex.as_str()).unwrap(),
            FeatureBits::EMPTY
        );

        for feature in Features::FLAGS {
            hex.clear();
            feature.value().bits().write_hex(&mut hex).unwrap();
            assert_eq!(
                FeatureBits::parse_hex(hex.as_str()).unwrap(),
                feature.value().bits(),
                "{hex}"
            );
        }
    }

    #[test]
    fn check_features_display() {
        use alloc::format;

        let feature = Features::CLEAR_TEXTURE;
        assert_eq!(format!("{feature}"), "CLEAR_TEXTURE");

        let feature = Features::CLEAR_TEXTURE | Features::BGRA8UNORM_STORAGE;
        assert_eq!(format!("{feature}"), "CLEAR_TEXTURE | BGRA8UNORM_STORAGE");
    }

    #[test]
    fn check_features_bits() {
        let bits = Features::all().bits();
        assert_eq!(Features::from_bits_retain(bits), Features::all());

        let bits = Features::empty().bits();
        assert_eq!(Features::from_bits_retain(bits), Features::empty());

        for feature in Features::FLAGS {
            let bits = feature.value().bits();
            assert_eq!(Features::from_bits_retain(bits), *feature.value());
        }

        let bits = Features::all().bits();
        assert_eq!(Features::from_bits_truncate(bits), Features::all());

        let bits = Features::empty().bits();
        assert_eq!(Features::from_bits_truncate(bits), Features::empty());

        for feature in Features::FLAGS {
            let bits = feature.value().bits();
            assert_eq!(Features::from_bits_truncate(bits), *feature.value());
        }

        let bits = Features::all().bits();
        assert_eq!(Features::from_bits(bits).unwrap(), Features::all());

        let bits = Features::empty().bits();
        assert_eq!(Features::from_bits(bits).unwrap(), Features::empty());

        for feature in Features::FLAGS {
            let bits = feature.value().bits();
            assert_eq!(Features::from_bits(bits).unwrap(), *feature.value());
        }
    }

    #[test]
    fn create_features_from_parts() {
        let features: Features = FeaturesWGPU::TEXTURE_ATOMIC.into();
        assert_eq!(features, Features::TEXTURE_ATOMIC);

        let features: Features = FeaturesWebGPU::TIMESTAMP_QUERY.into();
        assert_eq!(features, Features::TIMESTAMP_QUERY);

        let features: Features = Features::from(FeaturesWGPU::TEXTURE_ATOMIC)
            | Features::from(FeaturesWebGPU::TIMESTAMP_QUERY);
        assert_eq!(
            features,
            Features::TEXTURE_ATOMIC | Features::TIMESTAMP_QUERY
        );
        assert_eq!(
            features,
            Features::from_internal_flags(
                FeaturesWGPU::TEXTURE_ATOMIC,
                FeaturesWebGPU::TIMESTAMP_QUERY
            )
        );
    }
}
