use crate::{ImageClass, ImageDimension, ScalarKind, TypeInner, VectorSize};
use glsl::syntax::{BinaryOp, TypeSpecifierNonArray, UnaryOp};

pub fn glsl_to_spirv_unary_op(op: UnaryOp) -> crate::UnaryOperator {
    match op {
        UnaryOp::Inc => todo!(),
        UnaryOp::Dec => todo!(),
        UnaryOp::Add => todo!(),
        UnaryOp::Minus => crate::UnaryOperator::Negate,
        UnaryOp::Not => crate::UnaryOperator::Not,
        UnaryOp::Complement => todo!(),
    }
}

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

pub fn glsl_to_spirv_type(ty: TypeSpecifierNonArray) -> Option<TypeInner> {
    use TypeSpecifierNonArray::*;

    Some(match ty {
        Void => return None,
        Bool => TypeInner::Scalar {
            kind: ScalarKind::Bool,
            width: 1,
        },
        Int => TypeInner::Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        },
        UInt => TypeInner::Scalar {
            kind: ScalarKind::Uint,
            width: 4,
        },
        Float => TypeInner::Scalar {
            kind: ScalarKind::Float,
            width: 4,
        },
        Double => TypeInner::Scalar {
            kind: ScalarKind::Float,
            width: 8,
        },
        Vec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 4,
        },
        Vec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 4,
        },
        Vec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 4,
        },
        DVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 8,
        },
        DVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 8,
        },
        DVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 8,
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
            width: 4,
        },
        IVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Sint,
            width: 4,
        },
        IVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Sint,
            width: 4,
        },
        UVec2 => TypeInner::Vector {
            size: VectorSize::Bi,
            kind: ScalarKind::Uint,
            width: 4,
        },
        UVec3 => TypeInner::Vector {
            size: VectorSize::Tri,
            kind: ScalarKind::Uint,
            width: 4,
        },
        UVec4 => TypeInner::Vector {
            size: VectorSize::Quad,
            kind: ScalarKind::Uint,
            width: 4,
        },
        // Float Matrices
        Mat2 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat3 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat4 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat23 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat24 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat32 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat34 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat42 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 4,
        },
        Mat43 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 4,
        },
        // Double Matrices
        DMat2 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat3 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat4 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat23 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat24 => TypeInner::Matrix {
            columns: VectorSize::Bi,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat32 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat34 => TypeInner::Matrix {
            columns: VectorSize::Tri,
            rows: VectorSize::Quad,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat42 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Bi,
            kind: ScalarKind::Float,
            width: 8,
        },
        DMat43 => TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
            kind: ScalarKind::Float,
            width: 8,
        },
        TypeName(ty_name) => {
            if let Some(t_pos) = ty_name.0.find("texture") {
                let kind = match &ty_name.0[..t_pos] {
                    "" => ScalarKind::Float,
                    "i" => ScalarKind::Sint,
                    "u" => ScalarKind::Uint,
                    _ => panic!(),
                };
                let arrayed = ty_name.0.ends_with("Array");
                let (dim, class) = match &ty_name.0[(t_pos + 7)..] {
                    "1D" | "1DArray" => (ImageDimension::D1, ImageClass::Sampled),
                    "2D" | "2DArray" => (ImageDimension::D2, ImageClass::Sampled),
                    "3D" | "3DArray" => (ImageDimension::D3, ImageClass::Sampled),
                    "2DMS" | "2DMSArray" => (ImageDimension::D2, ImageClass::Multisampled),
                    "Cube" | "CubeArray" => (ImageDimension::Cube, ImageClass::Sampled),
                    _ => panic!(),
                };

                return Some(TypeInner::Image {
                    kind,
                    dim,
                    arrayed,
                    class,
                });
            }

            match ty_name.0.as_str() {
                "sampler" => TypeInner::Sampler { comparison: false },
                "samplerShadow" => TypeInner::Sampler { comparison: true },
                _ => unimplemented!(),
            }
        }
        _ => unimplemented!(),
    })
}
