//! Code shared between the WGSL front and back ends.

use core::fmt::{self, Display, Formatter};

use crate::diagnostic_filter::{
    FilterableTriggeringRule, Severity, StandardFilterableTriggeringRule,
};

impl Severity {
    const ERROR: &'static str = "error";
    const WARNING: &'static str = "warning";
    const INFO: &'static str = "info";
    const OFF: &'static str = "off";

    /// Convert from a sentinel word in WGSL into its associated [`Severity`], if possible.
    pub fn from_wgsl_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::ERROR => Self::Error,
            Self::WARNING => Self::Warning,
            Self::INFO => Self::Info,
            Self::OFF => Self::Off,
            _ => return None,
        })
    }
}

struct DisplayFilterableTriggeringRule<'a>(&'a FilterableTriggeringRule);

impl Display for DisplayFilterableTriggeringRule<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let &Self(inner) = self;
        match *inner {
            FilterableTriggeringRule::Standard(rule) => write!(f, "{}", rule.to_wgsl_ident()),
            FilterableTriggeringRule::Unknown(ref rule) => write!(f, "{rule}"),
            FilterableTriggeringRule::User(ref rules) => {
                let &[ref seg1, ref seg2] = rules.as_ref();
                write!(f, "{seg1}.{seg2}")
            }
        }
    }
}

impl FilterableTriggeringRule {
    /// [`Display`] this rule's identifiers in WGSL.
    pub const fn display_wgsl_ident(&self) -> impl Display + '_ {
        DisplayFilterableTriggeringRule(self)
    }
}

impl StandardFilterableTriggeringRule {
    const DERIVATIVE_UNIFORMITY: &'static str = "derivative_uniformity";

    /// Convert from a sentinel word in WGSL into its associated
    /// [`StandardFilterableTriggeringRule`], if possible.
    pub fn from_wgsl_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DERIVATIVE_UNIFORMITY => Self::DerivativeUniformity,
            _ => return None,
        })
    }

    /// Maps this [`StandardFilterableTriggeringRule`] into the sentinel word associated with it in
    /// WGSL.
    pub const fn to_wgsl_ident(self) -> &'static str {
        match self {
            Self::DerivativeUniformity => Self::DERIVATIVE_UNIFORMITY,
        }
    }
}

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
///   [`std::fmt::Display`] on some wrapper type instead.
pub trait ToWgsl: Sized {
    /// Return WGSL source code representation of `self`.
    fn to_wgsl(self) -> &'static str;
}

/// Types that may be able to return the WGSL source representation
/// for their values as a `'static' string.
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
///   [`std::fmt::Display`] on some wrapper type instead.
pub trait TryToWgsl: Sized {
    /// Return the WGSL form of `self` as a `'static` string.
    ///
    /// If `self` doesn't have a representation in WGSL (standard or
    /// as extended by Naga), then return `None`.
    fn try_to_wgsl(self) -> Option<&'static str>;

    /// What kind of WGSL thing `Self` represents.
    const DESCRIPTION: &'static str;
}

impl TryToWgsl for crate::MathFunction {
    const DESCRIPTION: &'static str = "math function";

    fn try_to_wgsl(self) -> Option<&'static str> {
        use crate::MathFunction as Mf;

        Some(match self {
            Mf::Abs => "abs",
            Mf::Min => "min",
            Mf::Max => "max",
            Mf::Clamp => "clamp",
            Mf::Saturate => "saturate",
            Mf::Cos => "cos",
            Mf::Cosh => "cosh",
            Mf::Sin => "sin",
            Mf::Sinh => "sinh",
            Mf::Tan => "tan",
            Mf::Tanh => "tanh",
            Mf::Acos => "acos",
            Mf::Asin => "asin",
            Mf::Atan => "atan",
            Mf::Atan2 => "atan2",
            Mf::Asinh => "asinh",
            Mf::Acosh => "acosh",
            Mf::Atanh => "atanh",
            Mf::Radians => "radians",
            Mf::Degrees => "degrees",
            Mf::Ceil => "ceil",
            Mf::Floor => "floor",
            Mf::Round => "round",
            Mf::Fract => "fract",
            Mf::Trunc => "trunc",
            Mf::Modf => "modf",
            Mf::Frexp => "frexp",
            Mf::Ldexp => "ldexp",
            Mf::Exp => "exp",
            Mf::Exp2 => "exp2",
            Mf::Log => "log",
            Mf::Log2 => "log2",
            Mf::Pow => "pow",
            Mf::Dot => "dot",
            Mf::Cross => "cross",
            Mf::Distance => "distance",
            Mf::Length => "length",
            Mf::Normalize => "normalize",
            Mf::FaceForward => "faceForward",
            Mf::Reflect => "reflect",
            Mf::Refract => "refract",
            Mf::Sign => "sign",
            Mf::Fma => "fma",
            Mf::Mix => "mix",
            Mf::Step => "step",
            Mf::SmoothStep => "smoothstep",
            Mf::Sqrt => "sqrt",
            Mf::InverseSqrt => "inverseSqrt",
            Mf::Transpose => "transpose",
            Mf::Determinant => "determinant",
            Mf::QuantizeToF16 => "quantizeToF16",
            Mf::CountTrailingZeros => "countTrailingZeros",
            Mf::CountLeadingZeros => "countLeadingZeros",
            Mf::CountOneBits => "countOneBits",
            Mf::ReverseBits => "reverseBits",
            Mf::ExtractBits => "extractBits",
            Mf::InsertBits => "insertBits",
            Mf::FirstTrailingBit => "firstTrailingBit",
            Mf::FirstLeadingBit => "firstLeadingBit",
            Mf::Pack4x8snorm => "pack4x8snorm",
            Mf::Pack4x8unorm => "pack4x8unorm",
            Mf::Pack2x16snorm => "pack2x16snorm",
            Mf::Pack2x16unorm => "pack2x16unorm",
            Mf::Pack2x16float => "pack2x16float",
            Mf::Pack4xI8 => "pack4xI8",
            Mf::Pack4xU8 => "pack4xU8",
            Mf::Unpack4x8snorm => "unpack4x8snorm",
            Mf::Unpack4x8unorm => "unpack4x8unorm",
            Mf::Unpack2x16snorm => "unpack2x16snorm",
            Mf::Unpack2x16unorm => "unpack2x16unorm",
            Mf::Unpack2x16float => "unpack2x16float",
            Mf::Unpack4xI8 => "unpack4xI8",
            Mf::Unpack4xU8 => "unpack4xU8",

            // Non-standard math functions.
            Mf::Inverse | Mf::Outer => return None,
        })
    }
}

impl TryToWgsl for crate::BuiltIn {
    const DESCRIPTION: &'static str = "builtin value";

    fn try_to_wgsl(self) -> Option<&'static str> {
        use crate::BuiltIn as Bi;
        Some(match self {
            Bi::Position { .. } => "position",
            Bi::ViewIndex => "view_index",
            Bi::InstanceIndex => "instance_index",
            Bi::VertexIndex => "vertex_index",
            Bi::FragDepth => "frag_depth",
            Bi::FrontFacing => "front_facing",
            Bi::PrimitiveIndex => "primitive_index",
            Bi::SampleIndex => "sample_index",
            Bi::SampleMask => "sample_mask",
            Bi::GlobalInvocationId => "global_invocation_id",
            Bi::LocalInvocationId => "local_invocation_id",
            Bi::LocalInvocationIndex => "local_invocation_index",
            Bi::WorkGroupId => "workgroup_id",
            Bi::NumWorkGroups => "num_workgroups",
            Bi::NumSubgroups => "num_subgroups",
            Bi::SubgroupId => "subgroup_id",
            Bi::SubgroupSize => "subgroup_size",
            Bi::SubgroupInvocationId => "subgroup_invocation_id",

            // Non-standard built-ins.
            Bi::BaseInstance
            | Bi::BaseVertex
            | Bi::ClipDistance
            | Bi::CullDistance
            | Bi::PointSize
            | Bi::DrawID
            | Bi::PointCoord
            | Bi::WorkGroupSize => return None,
        })
    }
}

impl ToWgsl for crate::Interpolation {
    fn to_wgsl(self) -> &'static str {
        match self {
            crate::Interpolation::Perspective => "perspective",
            crate::Interpolation::Linear => "linear",
            crate::Interpolation::Flat => "flat",
        }
    }
}

impl ToWgsl for crate::Sampling {
    fn to_wgsl(self) -> &'static str {
        match self {
            crate::Sampling::Center => "center",
            crate::Sampling::Centroid => "centroid",
            crate::Sampling::Sample => "sample",
            crate::Sampling::First => "first",
            crate::Sampling::Either => "either",
        }
    }
}

impl ToWgsl for crate::StorageFormat {
    fn to_wgsl(self) -> &'static str {
        use crate::StorageFormat as Sf;

        match self {
            Sf::R8Unorm => "r8unorm",
            Sf::R8Snorm => "r8snorm",
            Sf::R8Uint => "r8uint",
            Sf::R8Sint => "r8sint",
            Sf::R16Uint => "r16uint",
            Sf::R16Sint => "r16sint",
            Sf::R16Float => "r16float",
            Sf::Rg8Unorm => "rg8unorm",
            Sf::Rg8Snorm => "rg8snorm",
            Sf::Rg8Uint => "rg8uint",
            Sf::Rg8Sint => "rg8sint",
            Sf::R32Uint => "r32uint",
            Sf::R32Sint => "r32sint",
            Sf::R32Float => "r32float",
            Sf::Rg16Uint => "rg16uint",
            Sf::Rg16Sint => "rg16sint",
            Sf::Rg16Float => "rg16float",
            Sf::Rgba8Unorm => "rgba8unorm",
            Sf::Rgba8Snorm => "rgba8snorm",
            Sf::Rgba8Uint => "rgba8uint",
            Sf::Rgba8Sint => "rgba8sint",
            Sf::Bgra8Unorm => "bgra8unorm",
            Sf::Rgb10a2Uint => "rgb10a2uint",
            Sf::Rgb10a2Unorm => "rgb10a2unorm",
            Sf::Rg11b10Ufloat => "rg11b10float",
            Sf::R64Uint => "r64uint",
            Sf::Rg32Uint => "rg32uint",
            Sf::Rg32Sint => "rg32sint",
            Sf::Rg32Float => "rg32float",
            Sf::Rgba16Uint => "rgba16uint",
            Sf::Rgba16Sint => "rgba16sint",
            Sf::Rgba16Float => "rgba16float",
            Sf::Rgba32Uint => "rgba32uint",
            Sf::Rgba32Sint => "rgba32sint",
            Sf::Rgba32Float => "rgba32float",
            Sf::R16Unorm => "r16unorm",
            Sf::R16Snorm => "r16snorm",
            Sf::Rg16Unorm => "rg16unorm",
            Sf::Rg16Snorm => "rg16snorm",
            Sf::Rgba16Unorm => "rgba16unorm",
            Sf::Rgba16Snorm => "rgba16snorm",
        }
    }
}

impl TryToWgsl for crate::Scalar {
    const DESCRIPTION: &'static str = "scalar type";

    fn try_to_wgsl(self) -> Option<&'static str> {
        use crate::Scalar;

        Some(match self {
            Scalar::F64 => "f64",
            Scalar::F32 => "f32",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::I64 => "i64",
            Scalar::U64 => "u64",
            Scalar::BOOL => "bool",
            _ => return None,
        })
    }
}

impl ToWgsl for crate::ImageDimension {
    fn to_wgsl(self) -> &'static str {
        use crate::ImageDimension as IDim;

        match self {
            IDim::D1 => "1d",
            IDim::D2 => "2d",
            IDim::D3 => "3d",
            IDim::Cube => "cube",
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
