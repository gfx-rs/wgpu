use super::Error;

pub fn map_storage_class(word: &str) -> Result<crate::StorageClass, Error<'_>> {
    match word {
        "in" => Ok(crate::StorageClass::Input),
        "out" => Ok(crate::StorageClass::Output),
        "private" => Ok(crate::StorageClass::Private),
        "uniform" => Ok(crate::StorageClass::Uniform),
        "storage" => Ok(crate::StorageClass::Storage),
        _ => Err(Error::UnknownStorageClass(word)),
    }
}

pub fn map_built_in(word: &str) -> Result<crate::BuiltIn, Error<'_>> {
    Ok(match word {
        // vertex
        "position" => crate::BuiltIn::Position,
        "vertex_idx" => crate::BuiltIn::VertexIndex,
        "instance_idx" => crate::BuiltIn::InstanceIndex,
        // fragment
        "front_facing" => crate::BuiltIn::FrontFacing,
        "frag_coord" => crate::BuiltIn::FragCoord,
        "frag_depth" => crate::BuiltIn::FragDepth,
        // compute
        "global_invocation_id" => crate::BuiltIn::GlobalInvocationId,
        "local_invocation_id" => crate::BuiltIn::LocalInvocationId,
        "local_invocation_idx" => crate::BuiltIn::LocalInvocationIndex,
        _ => return Err(Error::UnknownBuiltin(word)),
    })
}

pub fn map_shader_stage(word: &str) -> Result<crate::ShaderStage, Error<'_>> {
    match word {
        "vertex" => Ok(crate::ShaderStage::Vertex),
        "fragment" => Ok(crate::ShaderStage::Fragment),
        "compute" => Ok(crate::ShaderStage::Compute),
        _ => Err(Error::UnknownShaderStage(word)),
    }
}

pub fn map_interpolation(word: &str) -> Result<crate::Interpolation, Error<'_>> {
    match word {
        "linear" => Ok(crate::Interpolation::Linear),
        "flat" => Ok(crate::Interpolation::Flat),
        "centroid" => Ok(crate::Interpolation::Centroid),
        "sample" => Ok(crate::Interpolation::Sample),
        "perspective" => Ok(crate::Interpolation::Perspective),
        _ => Err(Error::UnknownDecoration(word)),
    }
}

pub fn map_storage_format(word: &str) -> Result<crate::StorageFormat, Error<'_>> {
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
        _ => return Err(Error::UnknownStorageFormat(word)),
    })
}

pub fn get_scalar_type(word: &str) -> Option<(crate::ScalarKind, crate::Bytes)> {
    match word {
        "f32" => Some((crate::ScalarKind::Float, 4)),
        "i32" => Some((crate::ScalarKind::Sint, 4)),
        "u32" => Some((crate::ScalarKind::Uint, 4)),
        _ => None,
    }
}

pub fn get_intrinsic(word: &str) -> Option<crate::IntrinsicFunction> {
    match word {
        "any" => Some(crate::IntrinsicFunction::Any),
        "all" => Some(crate::IntrinsicFunction::All),
        "is_nan" => Some(crate::IntrinsicFunction::IsNan),
        "is_inf" => Some(crate::IntrinsicFunction::IsInf),
        "is_normal" => Some(crate::IntrinsicFunction::IsNormal),
        _ => None,
    }
}
pub fn get_derivative(word: &str) -> Option<crate::DerivativeAxis> {
    match word {
        "dpdx" => Some(crate::DerivativeAxis::X),
        "dpdy" => Some(crate::DerivativeAxis::Y),
        "dwidth" => Some(crate::DerivativeAxis::Width),
        _ => None,
    }
}
