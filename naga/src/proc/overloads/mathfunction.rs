//! Overload sets for [`ir::MathFunction`].

use crate::proc::overloads::any_overload_set::AnyOverloadSet;
use crate::proc::overloads::list::List;
use crate::proc::overloads::regular::regular;
use crate::proc::overloads::utils::{
    concrete_int_scalars, float_scalars, float_scalars_unimplemented_abstract, list, pairs, rule,
    scalar_or_vecn, triples, vector_sizes,
};
use crate::proc::overloads::OverloadSet;

use crate::ir;

impl ir::MathFunction {
    pub fn overloads(self) -> impl OverloadSet {
        

        let set: AnyOverloadSet = match self {
            // Component-wise unary numeric operations
            Self::Abs | Self::Sign => regular!(1, SCALAR|VECN of NUMERIC).into(),

            // Component-wise binary numeric operations
            Self::Min | Self::Max => regular!(2, SCALAR|VECN of NUMERIC).into(),

            // Component-wise ternary numeric operations
            Self::Clamp => regular!(3, SCALAR|VECN of NUMERIC).into(),

            // Component-wise unary floating-point operations
            Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Asinh
            | Self::Acosh
            | Self::Atanh
            | Self::Saturate
            | Self::Radians
            | Self::Degrees
            | Self::Ceil
            | Self::Floor
            | Self::Round
            | Self::Fract
            | Self::Trunc
            | Self::Exp
            | Self::Exp2
            | Self::Log
            | Self::Log2
            | Self::Sqrt
            | Self::InverseSqrt => regular!(1, SCALAR|VECN of FLOAT).into(),

            // Component-wise binary floating-point operations
            Self::Atan2 | Self::Pow | Self::Step => regular!(2, SCALAR|VECN of FLOAT).into(),

            // Component-wise ternary floating-point operations
            Self::Fma | Self::SmoothStep => regular!(3, SCALAR|VECN of FLOAT).into(),

            // Component-wise unary concrete integer operations
            Self::CountTrailingZeros
            | Self::CountLeadingZeros
            | Self::CountOneBits
            | Self::ReverseBits
            | Self::FirstTrailingBit
            | Self::FirstLeadingBit => regular!(1, SCALAR|VECN of CONCRETE_INTEGER).into(),

            // Packing functions
            Self::Pack4x8snorm | Self::Pack4x8unorm => regular!(1, VEC4 of F32 -> U32).into(),
            Self::Pack2x16snorm | Self::Pack2x16unorm | Self::Pack2x16float => {
                regular!(1, VEC2 of F32 -> U32).into()
            }
            Self::Pack4xI8 => regular!(1, VEC4 of I32 -> U32).into(),
            Self::Pack4xU8 => regular!(1, VEC4 of U32 -> U32).into(),
            Self::Pack4xI8Clamp => regular!(1, VEC4 of I32 -> U32).into(),
            Self::Pack4xU8Clamp => regular!(1, VEC4 of U32 -> U32).into(),

            // Unpacking functions
            Self::Unpack4x8snorm | Self::Unpack4x8unorm => regular!(1, SCALAR of U32 -> Vec4F).into(),
            Self::Unpack2x16snorm | Self::Unpack2x16unorm | Self::Unpack2x16float => {
                regular!(1, SCALAR of U32 -> Vec2F).into()
            }
            Self::Unpack4xI8 => regular!(1, SCALAR of U32 -> Vec4I).into(),
            Self::Unpack4xU8 => regular!(1, SCALAR of U32 -> Vec4U).into(),
            Self::Dot4I8Packed => regular!(2, SCALAR of U32 -> I32).into(),
            Self::Dot4U8Packed => regular!(2, SCALAR of U32 -> U32).into(),

            // One-off operations
            Self::Dot => regular!(2, VECN of NUMERIC -> Scalar).into(),
            Self::Modf => regular!(1, SCALAR|VECN of FLOAT_ABSTRACT_UNIMPLEMENTED -> Modf).into(),
            Self::Frexp => regular!(1, SCALAR|VECN of FLOAT_ABSTRACT_UNIMPLEMENTED -> Frexp).into(),
            Self::Ldexp => ldexp().into(),
            Self::Outer => outer().into(),
            Self::Cross => regular!(2, VEC3 of FLOAT).into(),
            Self::Distance => {
                regular!(2, SCALAR|VECN of FLOAT_ABSTRACT_UNIMPLEMENTED -> Scalar).into()
            }
            Self::Length => regular!(1, SCALAR|VECN of FLOAT_ABSTRACT_UNIMPLEMENTED -> Scalar).into(),
            Self::Normalize => regular!(1, VECN of FLOAT_ABSTRACT_UNIMPLEMENTED).into(),
            Self::FaceForward => regular!(3, VECN of FLOAT_ABSTRACT_UNIMPLEMENTED).into(),
            Self::Reflect => regular!(2, VECN of FLOAT_ABSTRACT_UNIMPLEMENTED).into(),
            Self::Refract => refract().into(),
            Self::Mix => mix().into(),
            Self::Inverse => regular!(1, MAT2X2|MAT3X3|MAT4X4 of FLOAT).into(),
            Self::Transpose => transpose().into(),
            Self::Determinant => regular!(1, MAT2X2|MAT3X3|MAT4X4 of FLOAT -> Scalar).into(),
            Self::QuantizeToF16 => regular!(1, SCALAR|VECN of F32).into(),
            Self::ExtractBits => extract_bits().into(),
            Self::InsertBits => insert_bits().into(),
        };

        set
    }
}

fn ldexp() -> List {
    /// Construct the exponent scalar given the mantissa's inner.
    fn exponent_from_mantissa(mantissa: ir::Scalar) -> ir::Scalar {
        match mantissa.kind {
            ir::ScalarKind::AbstractFloat => ir::Scalar::ABSTRACT_INT,
            ir::ScalarKind::Float => ir::Scalar::I32,
            _ => unreachable!("not a float scalar"),
        }
    }

    list(
        // The ldexp mantissa argument can be any floating-point type.
        float_scalars_unimplemented_abstract().flat_map(|mantissa_scalar| {
            // The exponent type is the integer counterpart of the mantissa type.
            let exponent_scalar = exponent_from_mantissa(mantissa_scalar);
            // There are scalar and vector component-wise overloads.
            scalar_or_vecn(mantissa_scalar)
                .zip(scalar_or_vecn(exponent_scalar))
                .map(move |(mantissa, exponent)| {
                    let result = mantissa.clone();
                    rule([mantissa, exponent], result)
                })
        }),
    )
}

fn outer() -> List {
    list(
        triples(
            vector_sizes(),
            vector_sizes(),
            float_scalars_unimplemented_abstract(),
        )
        .map(|(cols, rows, scalar)| {
            let left = ir::TypeInner::Vector { size: cols, scalar };
            let right = ir::TypeInner::Vector { size: rows, scalar };
            let result = ir::TypeInner::Matrix {
                columns: cols,
                rows,
                scalar,
            };
            rule([left, right], result)
        }),
    )
}

fn refract() -> List {
    list(
        pairs(vector_sizes(), float_scalars_unimplemented_abstract()).map(|(size, scalar)| {
            let incident = ir::TypeInner::Vector { size, scalar };
            let normal = incident.clone();
            let ratio = ir::TypeInner::Scalar(scalar);
            let result = incident.clone();
            rule([incident, normal, ratio], result)
        }),
    )
}

fn transpose() -> List {
    list(
        triples(vector_sizes(), vector_sizes(), float_scalars()).map(|(a, b, scalar)| {
            let input = ir::TypeInner::Matrix {
                columns: a,
                rows: b,
                scalar,
            };
            let output = ir::TypeInner::Matrix {
                columns: b,
                rows: a,
                scalar,
            };
            rule([input], output)
        }),
    )
}

fn extract_bits() -> List {
    list(concrete_int_scalars().flat_map(|scalar| {
        scalar_or_vecn(scalar).map(|input| {
            let offset = ir::TypeInner::Scalar(ir::Scalar::U32);
            let count = ir::TypeInner::Scalar(ir::Scalar::U32);
            let output = input.clone();
            rule([input, offset, count], output)
        })
    }))
}

fn insert_bits() -> List {
    list(concrete_int_scalars().flat_map(|scalar| {
        scalar_or_vecn(scalar).map(|input| {
            let newbits = input.clone();
            let offset = ir::TypeInner::Scalar(ir::Scalar::U32);
            let count = ir::TypeInner::Scalar(ir::Scalar::U32);
            let output = input.clone();
            rule([input, newbits, offset, count], output)
        })
    }))
}

fn mix() -> List {
    list(float_scalars().flat_map(|scalar| {
        scalar_or_vecn(scalar).flat_map(move |input| {
            let scalar_ratio = ir::TypeInner::Scalar(scalar);
            [
                rule([input.clone(), input.clone(), input.clone()], input.clone()),
                rule([input.clone(), input.clone(), scalar_ratio], input),
            ]
        })
    }))
}
