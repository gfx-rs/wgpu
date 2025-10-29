use crate::front::wgsl::parse::directive::enable_extension::{
    EnableExtensions, ImplementedEnableExtension,
};
use crate::front::wgsl::{Error, Result, Scalar};
use crate::Span;

use alloc::boxed::Box;

pub fn map_address_space(word: &str, span: Span) -> Result<'_, crate::AddressSpace> {
    match word {
        "private" => Ok(crate::AddressSpace::Private),
        "workgroup" => Ok(crate::AddressSpace::WorkGroup),
        "uniform" => Ok(crate::AddressSpace::Uniform),
        "storage" => Ok(crate::AddressSpace::Storage {
            access: crate::StorageAccess::default(),
        }),
        "push_constant" => Ok(crate::AddressSpace::PushConstant),
        "function" => Ok(crate::AddressSpace::Function),
        _ => Err(Box::new(Error::UnknownAddressSpace(span))),
    }
}

pub fn map_built_in(
    enable_extensions: &EnableExtensions,
    word: &str,
    span: Span,
) -> Result<'static, crate::BuiltIn> {
    let built_in = match word {
        "position" => crate::BuiltIn::Position { invariant: false },
        // vertex
        "vertex_index" => crate::BuiltIn::VertexIndex,
        "instance_index" => crate::BuiltIn::InstanceIndex,
        "view_index" => crate::BuiltIn::ViewIndex,
        "clip_distances" => crate::BuiltIn::ClipDistance,
        // fragment
        "front_facing" => crate::BuiltIn::FrontFacing,
        "frag_depth" => crate::BuiltIn::FragDepth,
        "primitive_index" => crate::BuiltIn::PrimitiveIndex,
        "barycentric" => crate::BuiltIn::Barycentric,
        "sample_index" => crate::BuiltIn::SampleIndex,
        "sample_mask" => crate::BuiltIn::SampleMask,
        // compute
        "global_invocation_id" => crate::BuiltIn::GlobalInvocationId,
        "local_invocation_id" => crate::BuiltIn::LocalInvocationId,
        "local_invocation_index" => crate::BuiltIn::LocalInvocationIndex,
        "workgroup_id" => crate::BuiltIn::WorkGroupId,
        "num_workgroups" => crate::BuiltIn::NumWorkGroups,
        // subgroup
        "num_subgroups" => crate::BuiltIn::NumSubgroups,
        "subgroup_id" => crate::BuiltIn::SubgroupId,
        "subgroup_size" => crate::BuiltIn::SubgroupSize,
        "subgroup_invocation_id" => crate::BuiltIn::SubgroupInvocationId,
        _ => return Err(Box::new(Error::UnknownBuiltin(span))),
    };
    match built_in {
        crate::BuiltIn::ClipDistance => {
            if !enable_extensions.contains(ImplementedEnableExtension::ClipDistances) {
                return Err(Box::new(Error::EnableExtensionNotEnabled {
                    span,
                    kind: ImplementedEnableExtension::ClipDistances.into(),
                }));
            }
        }
        _ => {}
    }
    Ok(built_in)
}

pub fn map_interpolation(word: &str, span: Span) -> Result<'_, crate::Interpolation> {
    match word {
        "linear" => Ok(crate::Interpolation::Linear),
        "flat" => Ok(crate::Interpolation::Flat),
        "perspective" => Ok(crate::Interpolation::Perspective),
        _ => Err(Box::new(Error::UnknownAttribute(span))),
    }
}

pub fn map_sampling(word: &str, span: Span) -> Result<'_, crate::Sampling> {
    match word {
        "center" => Ok(crate::Sampling::Center),
        "centroid" => Ok(crate::Sampling::Centroid),
        "sample" => Ok(crate::Sampling::Sample),
        "first" => Ok(crate::Sampling::First),
        "either" => Ok(crate::Sampling::Either),
        _ => Err(Box::new(Error::UnknownAttribute(span))),
    }
}

pub fn map_storage_format(word: &str, span: Span) -> Result<'_, crate::StorageFormat> {
    use crate::StorageFormat as Sf;
    Ok(match word {
        "r8unorm" => Sf::R8Unorm,
        "r8snorm" => Sf::R8Snorm,
        "r8uint" => Sf::R8Uint,
        "r8sint" => Sf::R8Sint,
        "r16unorm" => Sf::R16Unorm,
        "r16snorm" => Sf::R16Snorm,
        "r16uint" => Sf::R16Uint,
        "r16sint" => Sf::R16Sint,
        "r16float" => Sf::R16Float,
        "rg8unorm" => Sf::Rg8Unorm,
        "rg8snorm" => Sf::Rg8Snorm,
        "rg8uint" => Sf::Rg8Uint,
        "rg8sint" => Sf::Rg8Sint,
        "r32uint" => Sf::R32Uint,
        "r32sint" => Sf::R32Sint,
        "r32float" => Sf::R32Float,
        "rg16unorm" => Sf::Rg16Unorm,
        "rg16snorm" => Sf::Rg16Snorm,
        "rg16uint" => Sf::Rg16Uint,
        "rg16sint" => Sf::Rg16Sint,
        "rg16float" => Sf::Rg16Float,
        "rgba8unorm" => Sf::Rgba8Unorm,
        "rgba8snorm" => Sf::Rgba8Snorm,
        "rgba8uint" => Sf::Rgba8Uint,
        "rgba8sint" => Sf::Rgba8Sint,
        "rgb10a2uint" => Sf::Rgb10a2Uint,
        "rgb10a2unorm" => Sf::Rgb10a2Unorm,
        "rg11b10ufloat" => Sf::Rg11b10Ufloat,
        "r64uint" => Sf::R64Uint,
        "rg32uint" => Sf::Rg32Uint,
        "rg32sint" => Sf::Rg32Sint,
        "rg32float" => Sf::Rg32Float,
        "rgba16unorm" => Sf::Rgba16Unorm,
        "rgba16snorm" => Sf::Rgba16Snorm,
        "rgba16uint" => Sf::Rgba16Uint,
        "rgba16sint" => Sf::Rgba16Sint,
        "rgba16float" => Sf::Rgba16Float,
        "rgba32uint" => Sf::Rgba32Uint,
        "rgba32sint" => Sf::Rgba32Sint,
        "rgba32float" => Sf::Rgba32Float,
        "bgra8unorm" => Sf::Bgra8Unorm,
        _ => return Err(Box::new(Error::UnknownStorageFormat(span))),
    })
}

pub fn get_scalar_type(
    enable_extensions: &EnableExtensions,
    span: Span,
    word: &str,
) -> Result<'static, Option<Scalar>> {
    use crate::ScalarKind as Sk;
    let scalar = match word {
        "f16" => Some(Scalar {
            kind: Sk::Float,
            width: 2,
        }),
        "f32" => Some(Scalar {
            kind: Sk::Float,
            width: 4,
        }),
        "f64" => Some(Scalar {
            kind: Sk::Float,
            width: 8,
        }),
        "i32" => Some(Scalar {
            kind: Sk::Sint,
            width: 4,
        }),
        "u32" => Some(Scalar {
            kind: Sk::Uint,
            width: 4,
        }),
        "i64" => Some(Scalar {
            kind: Sk::Sint,
            width: 8,
        }),
        "u64" => Some(Scalar {
            kind: Sk::Uint,
            width: 8,
        }),
        "bool" => Some(Scalar {
            kind: Sk::Bool,
            width: crate::BOOL_WIDTH,
        }),
        _ => None,
    };

    if matches!(scalar, Some(Scalar::F16))
        && !enable_extensions.contains(ImplementedEnableExtension::F16)
    {
        return Err(Box::new(Error::EnableExtensionNotEnabled {
            span,
            kind: ImplementedEnableExtension::F16.into(),
        }));
    }

    Ok(scalar)
}

pub fn map_derivative(word: &str) -> Option<(crate::DerivativeAxis, crate::DerivativeControl)> {
    use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
    match word {
        "dpdxCoarse" => Some((Axis::X, Ctrl::Coarse)),
        "dpdyCoarse" => Some((Axis::Y, Ctrl::Coarse)),
        "fwidthCoarse" => Some((Axis::Width, Ctrl::Coarse)),
        "dpdxFine" => Some((Axis::X, Ctrl::Fine)),
        "dpdyFine" => Some((Axis::Y, Ctrl::Fine)),
        "fwidthFine" => Some((Axis::Width, Ctrl::Fine)),
        "dpdx" => Some((Axis::X, Ctrl::None)),
        "dpdy" => Some((Axis::Y, Ctrl::None)),
        "fwidth" => Some((Axis::Width, Ctrl::None)),
        _ => None,
    }
}

pub fn map_relational_fun(word: &str) -> Option<crate::RelationalFunction> {
    match word {
        "any" => Some(crate::RelationalFunction::Any),
        "all" => Some(crate::RelationalFunction::All),
        _ => None,
    }
}

pub fn map_standard_fun(word: &str) -> Option<crate::MathFunction> {
    use crate::MathFunction as Mf;
    Some(match word {
        // comparison
        "abs" => Mf::Abs,
        "min" => Mf::Min,
        "max" => Mf::Max,
        "clamp" => Mf::Clamp,
        "saturate" => Mf::Saturate,
        // trigonometry
        "cos" => Mf::Cos,
        "cosh" => Mf::Cosh,
        "sin" => Mf::Sin,
        "sinh" => Mf::Sinh,
        "tan" => Mf::Tan,
        "tanh" => Mf::Tanh,
        "acos" => Mf::Acos,
        "acosh" => Mf::Acosh,
        "asin" => Mf::Asin,
        "asinh" => Mf::Asinh,
        "atan" => Mf::Atan,
        "atanh" => Mf::Atanh,
        "atan2" => Mf::Atan2,
        "radians" => Mf::Radians,
        "degrees" => Mf::Degrees,
        // decomposition
        "ceil" => Mf::Ceil,
        "floor" => Mf::Floor,
        "round" => Mf::Round,
        "fract" => Mf::Fract,
        "trunc" => Mf::Trunc,
        "modf" => Mf::Modf,
        "frexp" => Mf::Frexp,
        "ldexp" => Mf::Ldexp,
        // exponent
        "exp" => Mf::Exp,
        "exp2" => Mf::Exp2,
        "log" => Mf::Log,
        "log2" => Mf::Log2,
        "pow" => Mf::Pow,
        // geometry
        "dot" => Mf::Dot,
        "dot4I8Packed" => Mf::Dot4I8Packed,
        "dot4U8Packed" => Mf::Dot4U8Packed,
        "cross" => Mf::Cross,
        "distance" => Mf::Distance,
        "length" => Mf::Length,
        "normalize" => Mf::Normalize,
        "faceForward" => Mf::FaceForward,
        "reflect" => Mf::Reflect,
        "refract" => Mf::Refract,
        // computational
        "sign" => Mf::Sign,
        "fma" => Mf::Fma,
        "mix" => Mf::Mix,
        "step" => Mf::Step,
        "smoothstep" => Mf::SmoothStep,
        "sqrt" => Mf::Sqrt,
        "inverseSqrt" => Mf::InverseSqrt,
        "transpose" => Mf::Transpose,
        "determinant" => Mf::Determinant,
        "quantizeToF16" => Mf::QuantizeToF16,
        // bits
        "countTrailingZeros" => Mf::CountTrailingZeros,
        "countLeadingZeros" => Mf::CountLeadingZeros,
        "countOneBits" => Mf::CountOneBits,
        "reverseBits" => Mf::ReverseBits,
        "extractBits" => Mf::ExtractBits,
        "insertBits" => Mf::InsertBits,
        "firstTrailingBit" => Mf::FirstTrailingBit,
        "firstLeadingBit" => Mf::FirstLeadingBit,
        // data packing
        "pack4x8snorm" => Mf::Pack4x8snorm,
        "pack4x8unorm" => Mf::Pack4x8unorm,
        "pack2x16snorm" => Mf::Pack2x16snorm,
        "pack2x16unorm" => Mf::Pack2x16unorm,
        "pack2x16float" => Mf::Pack2x16float,
        "pack4xI8" => Mf::Pack4xI8,
        "pack4xU8" => Mf::Pack4xU8,
        "pack4xI8Clamp" => Mf::Pack4xI8Clamp,
        "pack4xU8Clamp" => Mf::Pack4xU8Clamp,
        // data unpacking
        "unpack4x8snorm" => Mf::Unpack4x8snorm,
        "unpack4x8unorm" => Mf::Unpack4x8unorm,
        "unpack2x16snorm" => Mf::Unpack2x16snorm,
        "unpack2x16unorm" => Mf::Unpack2x16unorm,
        "unpack2x16float" => Mf::Unpack2x16float,
        "unpack4xI8" => Mf::Unpack4xI8,
        "unpack4xU8" => Mf::Unpack4xU8,
        _ => return None,
    })
}

pub fn map_conservative_depth(word: &str, span: Span) -> Result<'_, crate::ConservativeDepth> {
    use crate::ConservativeDepth as Cd;
    match word {
        "greater_equal" => Ok(Cd::GreaterEqual),
        "less_equal" => Ok(Cd::LessEqual),
        "unchanged" => Ok(Cd::Unchanged),
        _ => Err(Box::new(Error::UnknownConservativeDepth(span))),
    }
}

pub fn map_subgroup_operation(
    word: &str,
) -> Option<(crate::SubgroupOperation, crate::CollectiveOperation)> {
    use crate::CollectiveOperation as co;
    use crate::SubgroupOperation as sg;
    Some(match word {
        "subgroupAll" => (sg::All, co::Reduce),
        "subgroupAny" => (sg::Any, co::Reduce),
        "subgroupAdd" => (sg::Add, co::Reduce),
        "subgroupMul" => (sg::Mul, co::Reduce),
        "subgroupMin" => (sg::Min, co::Reduce),
        "subgroupMax" => (sg::Max, co::Reduce),
        "subgroupAnd" => (sg::And, co::Reduce),
        "subgroupOr" => (sg::Or, co::Reduce),
        "subgroupXor" => (sg::Xor, co::Reduce),
        "subgroupExclusiveAdd" => (sg::Add, co::ExclusiveScan),
        "subgroupExclusiveMul" => (sg::Mul, co::ExclusiveScan),
        "subgroupInclusiveAdd" => (sg::Add, co::InclusiveScan),
        "subgroupInclusiveMul" => (sg::Mul, co::InclusiveScan),
        _ => return None,
    })
}
