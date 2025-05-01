//! Generating WGSL source code for Naga IR types.

use alloc::format;
use alloc::string::{String, ToString};

/// Types that can return the WGSL source representation of their
/// values as a `'static` string.
///
/// This trait is specifically for types whose WGSL forms are simple
/// enough that they can always be returned as a static string.
///
/// - If only some values have a WGSL representation, consider
///   implementing [`TryToWgsl`] instead.
///
/// - If a type's WGSL form requires dynamic formatting, so that
///   returning a `&'static str` isn't feasible, consider implementing
///   [`core::fmt::Display`] on some wrapper type instead.
pub trait ToWgsl: Sized {
    /// Return WGSL source code representation of `self`.
    fn to_wgsl(self) -> &'static str;
}

/// Types that may be able to return the WGSL source representation
/// for their values as a `'static` string.
///
/// This trait is specifically for types whose values are either
/// simple enough that their WGSL form can be represented a static
/// string, or aren't representable in WGSL at all.
///
/// - If all values in the type have `&'static str` representations in
///   WGSL, consider implementing [`ToWgsl`] instead.
///
/// - If a type's WGSL form requires dynamic formatting, so that
///   returning a `&'static str` isn't feasible, consider implementing
///   [`core::fmt::Display`] on some wrapper type instead.
pub trait TryToWgsl: Sized {
    /// Return the WGSL form of `self` as a `'static` string.
    ///
    /// If `self` doesn't have a representation in WGSL (standard or
    /// as extended by Naga), then return `None`.
    fn try_to_wgsl(self) -> Option<&'static str>;

    /// What kind of WGSL thing `Self` represents.
    const DESCRIPTION: &'static str;

    /// Return the WGSL form of `self` as appropriate for diagnostics.
    ///
    /// If `self` can be expressed in WGSL, return that form as a
    /// [`String`]. Otherwise, return some representation of `self`
    /// that is appropriate for use in diagnostic messages.
    ///
    /// The default implementation of this function falls back to
    /// `self`'s [`Debug`] form.
    ///
    /// [`Debug`]: core::fmt::Debug
    fn to_wgsl_for_diagnostics(self) -> String
    where
        Self: core::fmt::Debug + Copy,
    {
        match self.try_to_wgsl() {
            Some(static_string) => static_string.to_string(),
            None => format!("{{non-WGSL {} {self:?}}}", Self::DESCRIPTION),
        }
    }
}

impl TryToWgsl for crate::MathFunction {
    const DESCRIPTION: &'static str = "math function";

    fn try_to_wgsl(self) -> Option<&'static str> {
        

        Some(match self {
            Self::Abs => "abs",
            Self::Min => "min",
            Self::Max => "max",
            Self::Clamp => "clamp",
            Self::Saturate => "saturate",
            Self::Cos => "cos",
            Self::Cosh => "cosh",
            Self::Sin => "sin",
            Self::Sinh => "sinh",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Acos => "acos",
            Self::Asin => "asin",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Radians => "radians",
            Self::Degrees => "degrees",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Round => "round",
            Self::Fract => "fract",
            Self::Trunc => "trunc",
            Self::Modf => "modf",
            Self::Frexp => "frexp",
            Self::Ldexp => "ldexp",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Pow => "pow",
            Self::Dot => "dot",
            Self::Dot4I8Packed => "dot4I8Packed",
            Self::Dot4U8Packed => "dot4U8Packed",
            Self::Cross => "cross",
            Self::Distance => "distance",
            Self::Length => "length",
            Self::Normalize => "normalize",
            Self::FaceForward => "faceForward",
            Self::Reflect => "reflect",
            Self::Refract => "refract",
            Self::Sign => "sign",
            Self::Fma => "fma",
            Self::Mix => "mix",
            Self::Step => "step",
            Self::SmoothStep => "smoothstep",
            Self::Sqrt => "sqrt",
            Self::InverseSqrt => "inverseSqrt",
            Self::Transpose => "transpose",
            Self::Determinant => "determinant",
            Self::QuantizeToF16 => "quantizeToF16",
            Self::CountTrailingZeros => "countTrailingZeros",
            Self::CountLeadingZeros => "countLeadingZeros",
            Self::CountOneBits => "countOneBits",
            Self::ReverseBits => "reverseBits",
            Self::ExtractBits => "extractBits",
            Self::InsertBits => "insertBits",
            Self::FirstTrailingBit => "firstTrailingBit",
            Self::FirstLeadingBit => "firstLeadingBit",
            Self::Pack4x8snorm => "pack4x8snorm",
            Self::Pack4x8unorm => "pack4x8unorm",
            Self::Pack2x16snorm => "pack2x16snorm",
            Self::Pack2x16unorm => "pack2x16unorm",
            Self::Pack2x16float => "pack2x16float",
            Self::Pack4xI8 => "pack4xI8",
            Self::Pack4xU8 => "pack4xU8",
            Self::Pack4xI8Clamp => "pack4xI8Clamp",
            Self::Pack4xU8Clamp => "pack4xU8Clamp",
            Self::Unpack4x8snorm => "unpack4x8snorm",
            Self::Unpack4x8unorm => "unpack4x8unorm",
            Self::Unpack2x16snorm => "unpack2x16snorm",
            Self::Unpack2x16unorm => "unpack2x16unorm",
            Self::Unpack2x16float => "unpack2x16float",
            Self::Unpack4xI8 => "unpack4xI8",
            Self::Unpack4xU8 => "unpack4xU8",

            // Non-standard math functions.
            Self::Inverse | Self::Outer => return None,
        })
    }
}

impl TryToWgsl for crate::BuiltIn {
    const DESCRIPTION: &'static str = "builtin value";

    fn try_to_wgsl(self) -> Option<&'static str> {
        
        Some(match self {
            Self::Position { .. } => "position",
            Self::ViewIndex => "view_index",
            Self::InstanceIndex => "instance_index",
            Self::VertexIndex => "vertex_index",
            Self::FragDepth => "frag_depth",
            Self::FrontFacing => "front_facing",
            Self::PrimitiveIndex => "primitive_index",
            Self::SampleIndex => "sample_index",
            Self::SampleMask => "sample_mask",
            Self::GlobalInvocationId => "global_invocation_id",
            Self::LocalInvocationId => "local_invocation_id",
            Self::LocalInvocationIndex => "local_invocation_index",
            Self::WorkGroupId => "workgroup_id",
            Self::NumWorkGroups => "num_workgroups",
            Self::NumSubgroups => "num_subgroups",
            Self::SubgroupId => "subgroup_id",
            Self::SubgroupSize => "subgroup_size",
            Self::SubgroupInvocationId => "subgroup_invocation_id",

            // Non-standard built-ins.
            Self::BaseInstance
            | Self::BaseVertex
            | Self::ClipDistance
            | Self::CullDistance
            | Self::PointSize
            | Self::DrawID
            | Self::PointCoord
            | Self::WorkGroupSize => return None,
        })
    }
}

impl ToWgsl for crate::Interpolation {
    fn to_wgsl(self) -> &'static str {
        match self {
            Self::Perspective => "perspective",
            Self::Linear => "linear",
            Self::Flat => "flat",
        }
    }
}

impl ToWgsl for crate::Sampling {
    fn to_wgsl(self) -> &'static str {
        match self {
            Self::Center => "center",
            Self::Centroid => "centroid",
            Self::Sample => "sample",
            Self::First => "first",
            Self::Either => "either",
        }
    }
}

impl ToWgsl for crate::StorageFormat {
    fn to_wgsl(self) -> &'static str {
        

        match self {
            Self::R8Unorm => "r8unorm",
            Self::R8Snorm => "r8snorm",
            Self::R8Uint => "r8uint",
            Self::R8Sint => "r8sint",
            Self::R16Uint => "r16uint",
            Self::R16Sint => "r16sint",
            Self::R16Float => "r16float",
            Self::Rg8Unorm => "rg8unorm",
            Self::Rg8Snorm => "rg8snorm",
            Self::Rg8Uint => "rg8uint",
            Self::Rg8Sint => "rg8sint",
            Self::R32Uint => "r32uint",
            Self::R32Sint => "r32sint",
            Self::R32Float => "r32float",
            Self::Rg16Uint => "rg16uint",
            Self::Rg16Sint => "rg16sint",
            Self::Rg16Float => "rg16float",
            Self::Rgba8Unorm => "rgba8unorm",
            Self::Rgba8Snorm => "rgba8snorm",
            Self::Rgba8Uint => "rgba8uint",
            Self::Rgba8Sint => "rgba8sint",
            Self::Bgra8Unorm => "bgra8unorm",
            Self::Rgb10a2Uint => "rgb10a2uint",
            Self::Rgb10a2Unorm => "rgb10a2unorm",
            Self::Rg11b10Ufloat => "rg11b10float",
            Self::R64Uint => "r64uint",
            Self::Rg32Uint => "rg32uint",
            Self::Rg32Sint => "rg32sint",
            Self::Rg32Float => "rg32float",
            Self::Rgba16Uint => "rgba16uint",
            Self::Rgba16Sint => "rgba16sint",
            Self::Rgba16Float => "rgba16float",
            Self::Rgba32Uint => "rgba32uint",
            Self::Rgba32Sint => "rgba32sint",
            Self::Rgba32Float => "rgba32float",
            Self::R16Unorm => "r16unorm",
            Self::R16Snorm => "r16snorm",
            Self::Rg16Unorm => "rg16unorm",
            Self::Rg16Snorm => "rg16snorm",
            Self::Rgba16Unorm => "rgba16unorm",
            Self::Rgba16Snorm => "rgba16snorm",
        }
    }
}

impl TryToWgsl for crate::Scalar {
    const DESCRIPTION: &'static str = "scalar type";

    fn try_to_wgsl(self) -> Option<&'static str> {
        

        Some(match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::BOOL => "bool",
            _ => return None,
        })
    }

    fn to_wgsl_for_diagnostics(self) -> String {
        match self.try_to_wgsl() {
            Some(static_string) => static_string.to_string(),
            None => match self.kind {
                crate::ScalarKind::Sint
                | crate::ScalarKind::Uint
                | crate::ScalarKind::Float
                | crate::ScalarKind::Bool => format!("{{non-WGSL scalar {self:?}}}"),
                crate::ScalarKind::AbstractInt => "{AbstractInt}".to_string(),
                crate::ScalarKind::AbstractFloat => "{AbstractFloat}".to_string(),
            },
        }
    }
}

impl ToWgsl for crate::ImageDimension {
    fn to_wgsl(self) -> &'static str {
        

        match self {
            Self::D1 => "1d",
            Self::D2 => "2d",
            Self::D3 => "3d",
            Self::Cube => "cube",
        }
    }
}

/// Return the WGSL address space and access mode strings for `space`.
///
/// Why don't we implement [`ToWgsl`] for [`AddressSpace`]?
///
/// In WGSL, the full form of a pointer type is `ptr<AS, T, AM>`, where:
/// - `AS` is the address space,
/// - `T` is the store type, and
/// - `AM` is the access mode.
///
/// Since the type `T` intervenes between the address space and the
/// access mode, there isn't really any individual WGSL grammar
/// production that corresponds to an [`AddressSpace`], so [`ToWgsl`]
/// is too simple-minded for this case.
///
/// Furthermore, we want to write `var<AS[, AM]>` for most address
/// spaces, but we want to just write `var foo: T` for handle types.
///
/// [`AddressSpace`]: crate::AddressSpace
pub const fn address_space_str(
    space: crate::AddressSpace,
) -> (Option<&'static str>, Option<&'static str>) {
    use crate::AddressSpace as As;

    (
        Some(match space {
            As::Private => "private",
            As::Uniform => "uniform",
            As::Storage { access } => {
                if access.contains(crate::StorageAccess::ATOMIC) {
                    return (Some("storage"), Some("atomic"));
                } else if access.contains(crate::StorageAccess::STORE) {
                    return (Some("storage"), Some("read_write"));
                } else {
                    "storage"
                }
            }
            As::PushConstant => "push_constant",
            As::WorkGroup => "workgroup",
            As::Handle => return (None, None),
            As::Function => "function",
        }),
        None,
    )
}
