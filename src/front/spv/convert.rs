use super::error::Error;
use num_traits::cast::FromPrimitive;
use std::convert::TryInto;

pub fn map_binary_operator(word: spirv::Op) -> Result<crate::BinaryOperator, Error> {
    use crate::BinaryOperator;
    use spirv::Op;

    match word {
        // Arithmetic Instructions +, -, *, /, %
        Op::IAdd | Op::FAdd => Ok(BinaryOperator::Add),
        Op::ISub | Op::FSub => Ok(BinaryOperator::Subtract),
        Op::IMul | Op::FMul => Ok(BinaryOperator::Multiply),
        Op::UDiv | Op::SDiv | Op::FDiv => Ok(BinaryOperator::Divide),
        Op::UMod | Op::SMod | Op::FMod => Ok(BinaryOperator::Modulo),
        // Relational and Logical Instructions
        Op::IEqual | Op::FOrdEqual | Op::FUnordEqual => Ok(BinaryOperator::Equal),
        Op::INotEqual | Op::FOrdNotEqual | Op::FUnordNotEqual => Ok(BinaryOperator::NotEqual),
        Op::ULessThan | Op::SLessThan | Op::FOrdLessThan | Op::FUnordLessThan => {
            Ok(BinaryOperator::Less)
        }
        Op::ULessThanEqual
        | Op::SLessThanEqual
        | Op::FOrdLessThanEqual
        | Op::FUnordLessThanEqual => Ok(BinaryOperator::LessEqual),
        Op::UGreaterThan | Op::SGreaterThan | Op::FOrdGreaterThan | Op::FUnordGreaterThan => {
            Ok(BinaryOperator::Greater)
        }
        Op::UGreaterThanEqual
        | Op::SGreaterThanEqual
        | Op::FOrdGreaterThanEqual
        | Op::FUnordGreaterThanEqual => Ok(BinaryOperator::GreaterEqual),
        _ => Err(Error::UnknownInstruction(word as u16)),
    }
}

pub fn map_vector_size(word: spirv::Word) -> Result<crate::VectorSize, Error> {
    match word {
        2 => Ok(crate::VectorSize::Bi),
        3 => Ok(crate::VectorSize::Tri),
        4 => Ok(crate::VectorSize::Quad),
        _ => Err(Error::InvalidVectorSize(word)),
    }
}

pub fn map_storage_class(word: spirv::Word) -> Result<crate::StorageClass, Error> {
    use spirv::StorageClass as Sc;
    match Sc::from_u32(word) {
        Some(Sc::UniformConstant) => Ok(crate::StorageClass::Constant),
        Some(Sc::Function) => Ok(crate::StorageClass::Function),
        Some(Sc::Input) => Ok(crate::StorageClass::Input),
        Some(Sc::Output) => Ok(crate::StorageClass::Output),
        Some(Sc::Private) => Ok(crate::StorageClass::Private),
        Some(Sc::StorageBuffer) => Ok(crate::StorageClass::StorageBuffer),
        Some(Sc::Uniform) => Ok(crate::StorageClass::Uniform),
        Some(Sc::Workgroup) => Ok(crate::StorageClass::WorkGroup),
        _ => Err(Error::UnsupportedStorageClass(word)),
    }
}

pub fn map_image_dim(word: spirv::Word) -> Result<crate::ImageDimension, Error> {
    use spirv::Dim as D;
    match D::from_u32(word) {
        Some(D::Dim1D) => Ok(crate::ImageDimension::D1),
        Some(D::Dim2D) => Ok(crate::ImageDimension::D2),
        Some(D::Dim3D) => Ok(crate::ImageDimension::D3),
        Some(D::DimCube) => Ok(crate::ImageDimension::Cube),
        _ => Err(Error::UnsupportedImageDim(word)),
    }
}

pub fn map_image_format(word: spirv::Word) -> Result<crate::StorageFormat, Error> {
    use spirv::ImageFormat as If;
    match If::from_u32(word) {
        Some(If::Rgba32f) => Ok(crate::StorageFormat::Rgba32f),
        _ => Err(Error::UnsupportedImageFormat(word)),
    }
}

pub fn map_width(word: spirv::Word) -> Result<crate::Bytes, Error> {
    (word >> 3) // bits to bytes
        .try_into()
        .map_err(|_| Error::InvalidTypeWidth(word))
}

pub fn map_builtin(word: spirv::Word) -> Result<crate::BuiltIn, Error> {
    use spirv::BuiltIn as Bi;
    Ok(match spirv::BuiltIn::from_u32(word) {
        Some(Bi::BaseInstance) => crate::BuiltIn::BaseInstance,
        Some(Bi::BaseVertex) => crate::BuiltIn::BaseVertex,
        Some(Bi::ClipDistance) => crate::BuiltIn::ClipDistance,
        Some(Bi::InstanceIndex) => crate::BuiltIn::InstanceIndex,
        Some(Bi::Position) => crate::BuiltIn::Position,
        Some(Bi::VertexIndex) => crate::BuiltIn::VertexIndex,
        // fragment
        Some(Bi::PointSize) => crate::BuiltIn::PointSize,
        Some(Bi::FragCoord) => crate::BuiltIn::FragCoord,
        Some(Bi::FrontFacing) => crate::BuiltIn::FrontFacing,
        Some(Bi::SampleId) => crate::BuiltIn::SampleIndex,
        Some(Bi::FragDepth) => crate::BuiltIn::FragDepth,
        // compute
        Some(Bi::GlobalInvocationId) => crate::BuiltIn::GlobalInvocationId,
        Some(Bi::LocalInvocationId) => crate::BuiltIn::LocalInvocationId,
        Some(Bi::LocalInvocationIndex) => crate::BuiltIn::LocalInvocationIndex,
        Some(Bi::WorkgroupId) => crate::BuiltIn::WorkGroupId,
        _ => return Err(Error::UnsupportedBuiltIn(word)),
    })
}
