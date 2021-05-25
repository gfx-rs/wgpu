use super::{Error, Span};

pub fn map_storage_class(word: &str, span: Span) -> Result<crate::StorageClass, Error<'_>> {
    match word {
        "private" => Ok(crate::StorageClass::Private),
        "workgroup" => Ok(crate::StorageClass::WorkGroup),
        "uniform" => Ok(crate::StorageClass::Uniform),
        "storage" => Ok(crate::StorageClass::Storage),
        "push_constant" => Ok(crate::StorageClass::PushConstant),
        _ => Err(Error::UnknownStorageClass(span)),
    }
}

pub fn map_built_in(word: &str, span: Span) -> Result<crate::BuiltIn, Error<'_>> {
    Ok(match word {
        "position" => crate::BuiltIn::Position,
        // vertex
        "vertex_index" => crate::BuiltIn::VertexIndex,
        "instance_index" => crate::BuiltIn::InstanceIndex,
        // fragment
        "front_facing" => crate::BuiltIn::FrontFacing,
        "frag_depth" => crate::BuiltIn::FragDepth,
        "sample_index" => crate::BuiltIn::SampleIndex,
        "sample_mask" => crate::BuiltIn::SampleMask,
        // compute
        "global_invocation_id" => crate::BuiltIn::GlobalInvocationId,
        "local_invocation_id" => crate::BuiltIn::LocalInvocationId,
        "local_invocation_index" => crate::BuiltIn::LocalInvocationIndex,
        "workgroup_id" => crate::BuiltIn::WorkGroupId,
        "workgroup_size" => crate::BuiltIn::WorkGroupSize,
        _ => return Err(Error::UnknownBuiltin(span)),
    })
}

pub fn map_shader_stage(word: &str, span: Span) -> Result<crate::ShaderStage, Error<'_>> {
    match word {
        "vertex" => Ok(crate::ShaderStage::Vertex),
        "fragment" => Ok(crate::ShaderStage::Fragment),
        "compute" => Ok(crate::ShaderStage::Compute),
        _ => Err(Error::UnknownShaderStage(span)),
    }
}

pub fn map_interpolation(word: &str, span: Span) -> Result<crate::Interpolation, Error<'_>> {
    match word {
        "linear" => Ok(crate::Interpolation::Linear),
        "flat" => Ok(crate::Interpolation::Flat),
        "perspective" => Ok(crate::Interpolation::Perspective),
        _ => Err(Error::UnknownAttribute(span)),
    }
}

pub fn map_sampling(word: &str, span: Span) -> Result<crate::Sampling, Error<'_>> {
    match word {
        "centroid" => Ok(crate::Sampling::Centroid),
        "sample" => Ok(crate::Sampling::Sample),
        _ => Err(Error::UnknownAttribute(span)),
    }
}

pub fn map_storage_format(word: &str, span: Span) -> Result<crate::StorageFormat, Error<'_>> {
    use crate::StorageFormat as Sf;
    Ok(match word {
        "r8unorm" => Sf::R8Unorm,
        "r8snorm" => Sf::R8Snorm,
        "r8uint" => Sf::R8Uint,
        "r8sint" => Sf::R8Sint,
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
        "rg16uint" => Sf::Rg16Uint,
        "rg16sint" => Sf::Rg16Sint,
        "rg16float" => Sf::Rg16Float,
        "rgba8unorm" => Sf::Rgba8Unorm,
        "rgba8snorm" => Sf::Rgba8Snorm,
        "rgba8uint" => Sf::Rgba8Uint,
        "rgba8sint" => Sf::Rgba8Sint,
        "rgb10a2unorm" => Sf::Rgb10a2Unorm,
        "rg11b10float" => Sf::Rg11b10Float,
        "rg32uint" => Sf::Rg32Uint,
        "rg32sint" => Sf::Rg32Sint,
        "rg32float" => Sf::Rg32Float,
        "rgba16uint" => Sf::Rgba16Uint,
        "rgba16sint" => Sf::Rgba16Sint,
        "rgba16float" => Sf::Rgba16Float,
        "rgba32uint" => Sf::Rgba32Uint,
        "rgba32sint" => Sf::Rgba32Sint,
        "rgba32float" => Sf::Rgba32Float,
        _ => return Err(Error::UnknownStorageFormat(span)),
    })
}

pub fn get_scalar_type(word: &str) -> Option<(crate::ScalarKind, crate::Bytes)> {
    match word {
        "f16" => Some((crate::ScalarKind::Float, 2)),
        "f32" => Some((crate::ScalarKind::Float, 4)),
        "f64" => Some((crate::ScalarKind::Float, 8)),
        "i8" => Some((crate::ScalarKind::Sint, 1)),
        "i16" => Some((crate::ScalarKind::Sint, 2)),
        "i32" => Some((crate::ScalarKind::Sint, 4)),
        "i64" => Some((crate::ScalarKind::Sint, 8)),
        "u8" => Some((crate::ScalarKind::Uint, 1)),
        "u16" => Some((crate::ScalarKind::Uint, 2)),
        "u32" => Some((crate::ScalarKind::Uint, 4)),
        "u64" => Some((crate::ScalarKind::Uint, 8)),
        "bool" => Some((crate::ScalarKind::Bool, crate::BOOL_WIDTH)),
        _ => None,
    }
}

pub fn map_derivative_axis(word: &str) -> Option<crate::DerivativeAxis> {
    match word {
        "dpdx" => Some(crate::DerivativeAxis::X),
        "dpdy" => Some(crate::DerivativeAxis::Y),
        "fwidth" => Some(crate::DerivativeAxis::Width),
        _ => None,
    }
}

pub fn map_relational_fun(word: &str) -> Option<crate::RelationalFunction> {
    match word {
        "any" => Some(crate::RelationalFunction::Any),
        "all" => Some(crate::RelationalFunction::All),
        "isFinite" => Some(crate::RelationalFunction::IsFinite),
        "isInf" => Some(crate::RelationalFunction::IsInf),
        "isNan" => Some(crate::RelationalFunction::IsNan),
        "isNormal" => Some(crate::RelationalFunction::IsNormal),
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
        // trigonometry
        "cos" => Mf::Cos,
        "cosh" => Mf::Cosh,
        "sin" => Mf::Sin,
        "sinh" => Mf::Sinh,
        "tan" => Mf::Tan,
        "tanh" => Mf::Tanh,
        "acos" => Mf::Acos,
        "asin" => Mf::Asin,
        "atan" => Mf::Atan,
        "atan2" => Mf::Atan2,
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
        "outerProduct" => Mf::Outer,
        "cross" => Mf::Cross,
        "distance" => Mf::Distance,
        "length" => Mf::Length,
        "normalize" => Mf::Normalize,
        "faceForward" => Mf::FaceForward,
        "reflect" => Mf::Reflect,
        // computational
        "sign" => Mf::Sign,
        "fma" => Mf::Fma,
        "mix" => Mf::Mix,
        "step" => Mf::Step,
        "smoothStep" => Mf::SmoothStep,
        "sqrt" => Mf::Sqrt,
        "inverseSqrt" => Mf::InverseSqrt,
        "transpose" => Mf::Transpose,
        "determinant" => Mf::Determinant,
        // bits
        "countOneBits" => Mf::CountOneBits,
        "reverseBits" => Mf::ReverseBits,
        _ => return None,
    })
}

pub fn map_conservative_depth(
    word: &str,
    span: Span,
) -> Result<crate::ConservativeDepth, Error<'_>> {
    use crate::ConservativeDepth as Cd;
    match word {
        "greater_equal" => Ok(Cd::GreaterEqual),
        "less_equal" => Ok(Cd::LessEqual),
        "unchanged" => Ok(Cd::Unchanged),
        _ => Err(Error::UnknownConservativeDepth(span)),
    }
}
