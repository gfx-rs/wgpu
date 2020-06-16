use crate::{Arena, ImageFlags, ScalarKind, Type, TypeInner, VectorSize};
use glsl::syntax::{BinaryOp, TypeSpecifierNonArray};
use spirv::Dim;

pub fn glsl_to_spirv_binary_op(op: BinaryOp) -> crate::BinaryOperator {
    match op {
        BinaryOp::Or => crate::BinaryOperator::LogicalOr,
        BinaryOp::Xor => todo!(),
        BinaryOp::And => crate::BinaryOperator::LogicalAnd,
        BinaryOp::BitOr => crate::BinaryOperator::InclusiveOr,
        BinaryOp::BitXor => crate::BinaryOperator::ExclusiveOr,
        BinaryOp::BitAnd => crate::BinaryOperator::And,
        BinaryOp::Equal => crate::BinaryOperator::Equal,
        BinaryOp::NonEqual => crate::BinaryOperator::NotEqual,
        BinaryOp::LT => crate::BinaryOperator::Less,
        BinaryOp::GT => crate::BinaryOperator::Greater,
        BinaryOp::LTE => crate::BinaryOperator::LessEqual,
        BinaryOp::GTE => crate::BinaryOperator::GreaterEqual,
        BinaryOp::LShift => crate::BinaryOperator::ShiftLeftLogical,
        BinaryOp::RShift => crate::BinaryOperator::ShiftRightArithmetic,
        BinaryOp::Add => crate::BinaryOperator::Add,
        BinaryOp::Sub => crate::BinaryOperator::Subtract,
        BinaryOp::Mult => crate::BinaryOperator::Multiply,
        BinaryOp::Div => crate::BinaryOperator::Divide,
        BinaryOp::Mod => crate::BinaryOperator::Modulo,
    }
}

pub fn glsl_to_spirv_type(ty: TypeSpecifierNonArray, types: &mut Arena<Type>) -> Option<TypeInner> {
    use TypeSpecifierNonArray::*;

    Some(match ty {
        Void => return None,
        Bool => TypeInner::Scalar {
            kind: ScalarKind::Bool,
            width: 1,
        },
        Int => TypeInner::Scalar {
            kind: ScalarKind::Sint,
            width: 32,
        },
        UInt => TypeInner::Scalar {
            kind: ScalarKind::Uint,
            width: 32,
        },
        Float => TypeInner::Scalar {
            kind: ScalarKind::Float,
            width: 32,
        },
        Double => TypeInner::Scalar {
            kind: ScalarKind::Float,
            width: 64,
        },
        Vec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 32,
        },
        Vec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 32,
        },
        Vec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 32,
        },
        DVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 64,
        },
        DVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 64,
        },
        DVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 64,
        },
        BVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Bool,
            width: 1,
        },
        BVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Bool,
            width: 1,
        },
        BVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Bool,
            width: 1,
        },
        IVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Sint,
            width: 32,
        },
        IVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Sint,
            width: 32,
        },
        IVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Sint,
            width: 32,
        },
        UVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Uint,
            width: 32,
        },
        UVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Uint,
            width: 32,
        },
        UVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Uint,
            width: 32,
        },
        // Float Matrices
        Mat2 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat3 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat4 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat23 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat24 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat32 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat34 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat42 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 32,
        },
        Mat43 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 32,
        },
        // Double Matrices
        DMat2 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat3 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat4 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat23 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat24 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat32 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat34 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat42 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 64,
        },
        DMat43 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 64,
        },
        TypeName(ty_name) => {
            if let Some(t_pos) = ty_name.0.find("texture") {
                let scalar_kind = match &ty_name.0[..t_pos] {
                    "" => ScalarKind::Float,
                    "i" => ScalarKind::Sint,
                    "u" => ScalarKind::Uint,
                    _ => panic!(),
                };
                let base = types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: scalar_kind,
                        width: 32,
                    },
                });

                let (dim, flags) = match &ty_name.0[(t_pos + 7)..] {
                    "1D" => (Dim::Dim1D, ImageFlags::SAMPLED),
                    "2D" => (Dim::Dim2D, ImageFlags::SAMPLED),
                    "3D" => (Dim::Dim3D, ImageFlags::SAMPLED),
                    "1DArray" => (Dim::Dim1D, ImageFlags::SAMPLED | ImageFlags::ARRAYED),
                    "2DArray" => (Dim::Dim2D, ImageFlags::SAMPLED | ImageFlags::ARRAYED),
                    "3DArray" => (Dim::Dim3D, ImageFlags::SAMPLED | ImageFlags::ARRAYED),
                    "2DMS" => (Dim::Dim2D, ImageFlags::SAMPLED | ImageFlags::MULTISAMPLED),
                    "2DMSArray" => (
                        Dim::Dim2D,
                        ImageFlags::SAMPLED | ImageFlags::ARRAYED | ImageFlags::MULTISAMPLED,
                    ),
                    "2DRect" => (Dim::DimRect, ImageFlags::SAMPLED),
                    "Cube" => (Dim::DimCube, ImageFlags::SAMPLED),
                    "CubeArray" => (Dim::DimCube, ImageFlags::SAMPLED | ImageFlags::ARRAYED),
                    "Buffer" => (Dim::DimBuffer, ImageFlags::SAMPLED),
                    _ => panic!(),
                };

                return Some(TypeInner::Image { base, dim, flags });
            }

            if ty_name.0 == "sampler" {
                return Some(TypeInner::Sampler { comparison: false }); //TODO
            }
            unimplemented!()
        }
        _ => unimplemented!(),
    })
}
