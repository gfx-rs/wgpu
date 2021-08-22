use super::{
    ast::{FunctionDeclaration, FunctionKind, Overload, ParameterInfo, ParameterQualifier},
    context::Context,
    Error, ErrorKind, Parser, Result, SourceMetadata,
};
use crate::{
    BinaryOperator, Block, Constant, ConstantInner, DerivativeAxis, Expression, Handle, ImageClass,
    ImageDimension, ImageQuery, MathFunction, Module, RelationalFunction, SampleLevel,
    ScalarKind as Sk, ScalarValue, Span, Type, TypeInner, VectorSize,
};

impl Module {
    /// Helper function, to create a function prototype for a builtin
    fn add_builtin(&mut self, args: Vec<TypeInner>, builtin: MacroCall) -> Overload {
        let mut parameters = Vec::with_capacity(args.len());
        let mut parameters_info = Vec::with_capacity(args.len());

        for arg in args {
            parameters.push(self.types.fetch_or_append(
                Type {
                    name: None,
                    inner: arg,
                },
                Span::Unknown,
            ));
            parameters_info.push(ParameterInfo {
                qualifier: ParameterQualifier::In,
                depth: false,
            });
        }

        Overload {
            parameters,
            parameters_info,
            kind: FunctionKind::Macro(builtin),
            defined: false,
            void: false,
        }
    }
}

/// Inject builtins into
///
/// This is done to not add a large startup cost and not increase memory
/// usage if it isn't needed.
///
/// This version does not add builtins with arguments using the double type
/// [`inject_double_builtin`](inject_double_builtin) for builtins
/// using the double type
pub fn inject_builtin(declaration: &mut FunctionDeclaration, module: &mut Module, name: &str) {
    use crate::ImageDimension as Dim;

    declaration.builtin = true;
    let width = 4;
    match name {
        "sampler1D" | "sampler1DArray" | "sampler2D" | "sampler2DArray" | "sampler2DMS"
        | "sampler2DMSArray" | "sampler3D" | "samplerCube" | "samplerCubeArray" => {
            declaration.overloads.push(module.add_builtin(
                vec![
                    TypeInner::Image {
                        dim: match name {
                            "sampler1D" | "sampler1DArray" => Dim::D1,
                            "sampler2D" | "sampler2DArray" | "sampler2DMS" | "sampler2DMSArray" => {
                                Dim::D2
                            }
                            "sampler3D" => Dim::D3,
                            _ => Dim::Cube,
                        },
                        arrayed: matches!(
                            name,
                            "sampler1DArray"
                                | "sampler2DArray"
                                | "sampler2DMSArray"
                                | "samplerCubeArray"
                        ),
                        class: ImageClass::Sampled {
                            kind: Sk::Float,
                            multi: matches!(name, "sampler2DMS" | "sampler2DMSArray"),
                        },
                    },
                    TypeInner::Sampler { comparison: false },
                ],
                MacroCall::Sampler,
            ))
        }
        "sampler1DShadow"
        | "sampler1DArrayShadow"
        | "sampler2DShadow"
        | "sampler2DArrayShadow"
        | "samplerCubeShadow"
        | "samplerCubeArrayShadow" => {
            let dim = match name {
                "sampler1DShadow" | "sampler1DArrayShadow" => Dim::D1,
                "sampler2DShadow" | "sampler2DArrayShadow" => Dim::D2,
                _ => Dim::Cube,
            };
            let arrayed = matches!(
                name,
                "sampler1DArrayShadow" | "sampler2DArrayShadow" | "samplerCubeArrayShadow"
            );

            for i in 0..2 {
                let ty = TypeInner::Image {
                    dim,
                    arrayed,
                    class: match i {
                        0 => ImageClass::Sampled {
                            kind: Sk::Float,
                            multi: false,
                        },
                        _ => ImageClass::Depth { multi: false },
                    },
                };

                declaration.overloads.push(module.add_builtin(
                    vec![ty, TypeInner::Sampler { comparison: true }],
                    MacroCall::SamplerShadow,
                ))
            }
        }
        "texture" | "textureLod" => {
            // bits layout
            // bits 0 trough 1 - dims
            // bit 2 - shadow
            // bit 3 - array
            // bit 4 - !bias
            //
            // 0b11111 is the latest since there are not 3D arrayed shadowed
            // textures and cube arrayed shadows textures always have a
            // compare argument
            for bits in 0..(0b100000) {
                let dim = bits & 0b11;
                let shadow = bits & 0b100 == 0b100;
                let arrayed = bits & 0b1000 == 0b1000;
                let bias = bits & 0b10000 == 0b00000;
                let builtin = match name {
                    "texture" => MacroCall::Texture,
                    _ => MacroCall::TextureLod,
                };

                // Shadow, arrayed or both 3D images are not allowed
                if (shadow || arrayed) && dim == 0b11
                    || (builtin == MacroCall::TextureLod
                        && (dim != 0b00 && shadow && arrayed || dim == 0b10 && shadow || !bias))
                {
                    continue;
                }

                let class = match shadow {
                    true => ImageClass::Depth { multi: false },
                    false => ImageClass::Sampled {
                        kind: Sk::Float,
                        multi: false,
                    },
                };

                let image = TypeInner::Image {
                    dim: match dim {
                        0b00 => Dim::D1,
                        0b01 => Dim::D2,
                        0b10 => Dim::Cube,
                        _ => Dim::D3,
                    },
                    arrayed,
                    class,
                };

                let vector = match (dim, shadow) {
                    (0b00, false) => TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    },
                    (0b00, true) | (0b11, _) => TypeInner::Vector {
                        size: VectorSize::Tri,
                        kind: Sk::Float,
                        width,
                    },
                    (_, _) => {
                        let size = match dim + arrayed as u32 + shadow as u32 {
                            1 => VectorSize::Bi,
                            2 => VectorSize::Tri,
                            _ => VectorSize::Quad,
                        };

                        TypeInner::Vector {
                            size,
                            kind: Sk::Float,
                            width,
                        }
                    }
                };

                let mut args = vec![image, vector];

                if bias {
                    args.push(TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    })
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, builtin))
            }
        }
        "textureProj" => {
            // bits layout
            // bit 0 - shadow
            // bit 1 - bias
            // bits 2 trough 3 - dims
            //
            // 0b0111 is the latest since there are only 1D and 2D shadow
            // variants
            for bits in 0..(0b1000) {
                let dim = bits >> 2;
                let shadow = bits & 0b1 == 0b1;
                let bias = bits & 0b10 == 0b10;

                let class = match shadow {
                    true => ImageClass::Depth { multi: false },
                    false => ImageClass::Sampled {
                        kind: Sk::Float,
                        multi: false,
                    },
                };

                let image = TypeInner::Image {
                    dim: match dim {
                        0b00 => Dim::D1,
                        0b01 => Dim::D2,
                        _ => Dim::Cube,
                    },
                    arrayed: false,
                    class,
                };

                let vector = TypeInner::Vector {
                    size: VectorSize::Quad,
                    kind: Sk::Float,
                    width,
                };

                let mut args = vec![image, vector];

                if bias {
                    args.push(TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    })
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::TextureProj))
            }
        }
        "textureGrad" => {
            // bits layout
            // bits 0 trough 1 - dims
            // bit 2 - shadow
            // bit 3 - array
            //
            // 0b1010 is the latest since there are no 3D arrayed shadowed
            // textures and cube arrayed shadows textures are not allowed
            for bits in 0..(0b1011) {
                let dim = bits & 0b11;
                let shadow = bits & 0b100 == 0b100;
                let arrayed = bits & 0b1000 == 0b1000;

                // Shadow, arrayed or both 3D images are not allowed
                if shadow  && dim == 0b11
                        // samplerCubeArrayShadow is not allowed
                        || shadow && arrayed && dim == 0b10
                {
                    continue;
                }

                let class = match shadow {
                    true => ImageClass::Depth { multi: false },
                    false => ImageClass::Sampled {
                        kind: Sk::Float,
                        multi: false,
                    },
                };

                let image = TypeInner::Image {
                    dim: match dim {
                        0b00 => Dim::D1,
                        0b01 => Dim::D2,
                        0b10 => Dim::Cube,
                        _ => Dim::D3,
                    },
                    arrayed,
                    class,
                };

                let vector = match (dim, shadow) {
                    (0b00, false) => TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    },
                    (0b00, true) | (0b11, _) => TypeInner::Vector {
                        size: VectorSize::Tri,
                        kind: Sk::Float,
                        width,
                    },
                    (_, _) => {
                        let size = match dim + arrayed as u32 + shadow as u32 {
                            1 => VectorSize::Bi,
                            2 => VectorSize::Tri,
                            _ => VectorSize::Quad,
                        };

                        TypeInner::Vector {
                            size,
                            kind: Sk::Float,
                            width,
                        }
                    }
                };

                let size = match dim {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    _ => Some(VectorSize::Tri),
                };

                let ty = || match size {
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    },
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width,
                    },
                };

                let args = vec![image, vector, ty(), ty()];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::TextureGrad(shadow)))
            }
        }
        "textureSize" => {
            // bits layout
            // bits 0 trough 1 - dims
            // bit 2 - shadow
            // bit 3 - array
            for bits in 0..(0b10000) {
                let dim = bits & 0b11;
                let shadow = bits & 0b100 == 0b100;
                let arrayed = bits & 0b1000 == 0b1000;

                // Shadow, arrayed or both 3D images are not allowed
                if (shadow || arrayed) && dim == 0b11 {
                    continue;
                }

                let class = match shadow {
                    true => ImageClass::Depth { multi: false },
                    false => ImageClass::Sampled {
                        kind: Sk::Float,
                        multi: false,
                    },
                };

                let image = TypeInner::Image {
                    dim: match dim {
                        0b00 => Dim::D1,
                        0b01 => Dim::D2,
                        0b10 => Dim::Cube,
                        _ => Dim::D3,
                    },
                    arrayed,
                    class,
                };

                let args = vec![
                    image,
                    TypeInner::Scalar {
                        kind: Sk::Sint,
                        width,
                    },
                ];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::TextureSize))
            }
        }
        "texelFetch" => {
            // bits layout
            // bit 0 - dim part 1 - 1D/2D
            // bit 1 - array
            // bit 2 - dim part 2 - 3D
            //
            // 0b100 is the latest since 3D arrayed images aren't allowed
            for bits in 0..(0b101) {
                let dim = bits & 0b1 | (bits & 0b100) >> 1;
                let arrayed = bits & 0b1000 == 0b1000;

                let image = TypeInner::Image {
                    dim: match dim {
                        0b00 => Dim::D1,
                        0b01 => Dim::D2,
                        _ => Dim::D3,
                    },
                    arrayed,
                    class: ImageClass::Sampled {
                        kind: Sk::Float,
                        multi: false,
                    },
                };

                let vector = match (dim, arrayed) {
                    (0b00, false) => TypeInner::Scalar {
                        kind: Sk::Sint,
                        width,
                    },
                    (_, _) => {
                        let size = match dim + arrayed as u32 {
                            1 => VectorSize::Bi,
                            2 => VectorSize::Tri,
                            _ => VectorSize::Quad,
                        };

                        TypeInner::Vector {
                            size,
                            kind: Sk::Sint,
                            width,
                        }
                    }
                };

                let args = vec![
                    image,
                    vector,
                    TypeInner::Scalar {
                        kind: Sk::Sint,
                        width,
                    },
                ];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::TexelFetch))
            }
        }
        "sin" | "exp" | "exp2" | "sinh" | "cos" | "cosh" | "tan" | "tanh" | "acos" | "asin"
        | "log" | "log2" | "radians" | "degrees" | "asinh" | "acosh" | "atanh"
        | "floatBitsToInt" | "floatBitsToUint" | "dFdx" | "dFdxFine" | "dFdxCoarse" | "dFdy"
        | "dFdyFine" | "dFdyCoarse" | "fwidth" | "fwidthFine" | "fwidthCoarse" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = Sk::Float;

                declaration.overloads.push(module.add_builtin(
                    vec![match size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    }],
                    match name {
                        "sin" => MacroCall::MathFunction(MathFunction::Sin),
                        "exp" => MacroCall::MathFunction(MathFunction::Exp),
                        "exp2" => MacroCall::MathFunction(MathFunction::Exp2),
                        "sinh" => MacroCall::MathFunction(MathFunction::Sinh),
                        "cos" => MacroCall::MathFunction(MathFunction::Cos),
                        "cosh" => MacroCall::MathFunction(MathFunction::Cosh),
                        "tan" => MacroCall::MathFunction(MathFunction::Tan),
                        "tanh" => MacroCall::MathFunction(MathFunction::Tanh),
                        "acos" => MacroCall::MathFunction(MathFunction::Acos),
                        "asin" => MacroCall::MathFunction(MathFunction::Asin),
                        "log" => MacroCall::MathFunction(MathFunction::Log),
                        "log2" => MacroCall::MathFunction(MathFunction::Log2),
                        "asinh" => MacroCall::MathFunction(MathFunction::Asinh),
                        "acosh" => MacroCall::MathFunction(MathFunction::Acosh),
                        "atanh" => MacroCall::MathFunction(MathFunction::Atanh),
                        "radians" => MacroCall::ConstMultiply(std::f64::consts::PI / 180.0),
                        "degrees" => MacroCall::ConstMultiply(180.0 / std::f64::consts::PI),
                        "floatBitsToInt" => MacroCall::BitCast(Sk::Sint),
                        "floatBitsToUint" => MacroCall::BitCast(Sk::Uint),
                        "dFdx" | "dFdxFine" | "dFdxCoarse" => {
                            MacroCall::Derivate(DerivativeAxis::X)
                        }
                        "dFdy" | "dFdyFine" | "dFdyCoarse" => {
                            MacroCall::Derivate(DerivativeAxis::Y)
                        }
                        "fwidth" | "fwidthFine" | "fwidthCoarse" => {
                            MacroCall::Derivate(DerivativeAxis::Width)
                        }
                        _ => unreachable!(),
                    },
                ))
            }
        }
        "intBitsToFloat" | "uintBitsToFloat" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = match name {
                    "intBitsToFloat" => Sk::Sint,
                    _ => Sk::Uint,
                };

                declaration.overloads.push(module.add_builtin(
                    vec![match size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    }],
                    MacroCall::BitCast(Sk::Float),
                ))
            }
        }
        "pow" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = Sk::Float;
                let ty = || match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };

                declaration.overloads.push(
                    module
                        .add_builtin(vec![ty(), ty()], MacroCall::MathFunction(MathFunction::Pow)),
                )
            }
        }
        "abs" | "sign" => {
            // bits layout
            // bit 0 trough 1 - dims
            // bit 2 - float/sint
            for bits in 0..(0b1000) {
                let size = match bits & 0b11 {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = match bits >> 2 {
                    0b0 => Sk::Float,
                    _ => Sk::Sint,
                };

                let args = vec![match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                }];

                declaration.overloads.push(module.add_builtin(
                    args,
                    MacroCall::MathFunction(match name {
                        "abs" => MathFunction::Abs,
                        "sign" => MathFunction::Sign,
                        _ => unreachable!(),
                    }),
                ))
            }
        }
        "bitCount" | "bitfieldReverse" => {
            // bits layout
            // bit 0 - int/uint
            // bit 1 trough 2 - dims
            for bits in 0..(0b1000) {
                let kind = match bits & 0b1 {
                    0b0 => Sk::Sint,
                    _ => Sk::Uint,
                };
                let size = match bits >> 1 {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                let args = vec![match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                }];

                declaration.overloads.push(module.add_builtin(
                    args,
                    MacroCall::MathFunction(match name {
                        "bitCount" => MathFunction::CountOneBits,
                        "bitfieldReverse" => MathFunction::ReverseBits,
                        _ => unreachable!(),
                    }),
                ))
            }
        }
        "atan" => {
            // bits layout
            // bit 0 - atan/atan2
            // bit 1 trough 2 - dims
            for bits in 0..(0b1000) {
                let fun = match bits & 0b1 {
                    0b0 => MathFunction::Atan,
                    _ => MathFunction::Atan2,
                };
                let size = match bits >> 1 {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = Sk::Float;
                let ty = || match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };

                let mut args = vec![ty()];

                if fun == MathFunction::Atan2 {
                    args.push(ty())
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::MathFunction(fun)))
            }
        }
        "all" | "any" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b11) {
                let size = match bits {
                    0b00 => VectorSize::Bi,
                    0b01 => VectorSize::Tri,
                    _ => VectorSize::Quad,
                };

                let args = vec![TypeInner::Vector {
                    size,
                    kind: Sk::Bool,
                    width: crate::BOOL_WIDTH,
                }];

                let fun = MacroCall::Relational(match name {
                    "all" => RelationalFunction::All,
                    "any" => RelationalFunction::Any,
                    _ => unreachable!(),
                });

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
        | "notEqual" => {
            for bits in 0..(0b1001) {
                let (size, kind) = match bits {
                    0b0000 => (VectorSize::Bi, Sk::Float),
                    0b0001 => (VectorSize::Tri, Sk::Float),
                    0b0010 => (VectorSize::Quad, Sk::Float),
                    0b0011 => (VectorSize::Bi, Sk::Sint),
                    0b0100 => (VectorSize::Tri, Sk::Sint),
                    0b0101 => (VectorSize::Quad, Sk::Sint),
                    0b0110 => (VectorSize::Bi, Sk::Uint),
                    0b0111 => (VectorSize::Tri, Sk::Uint),
                    _ => (VectorSize::Quad, Sk::Uint),
                };

                let ty = || TypeInner::Vector { size, kind, width };
                let args = vec![ty(), ty()];

                let fun = MacroCall::Binary(match name {
                    "lessThan" => BinaryOperator::Less,
                    "greaterThan" => BinaryOperator::Greater,
                    "lessThanEqual" => BinaryOperator::LessEqual,
                    "greaterThanEqual" => BinaryOperator::GreaterEqual,
                    "equal" => BinaryOperator::Equal,
                    "notEqual" => BinaryOperator::NotEqual,
                    _ => unreachable!(),
                });

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "min" | "max" => {
            // bits layout
            // bit 0 trough 1 - scalar kind
            // bit 2 trough 4 - dims
            for bits in 0..(0b11100) {
                let kind = match bits & 0b11 {
                    0b00 => Sk::Float,
                    0b01 => Sk::Sint,
                    0b10 => Sk::Uint,
                    _ => continue,
                };
                let (size, second_size) = match bits >> 2 {
                    0b000 => (None, None),
                    0b001 => (Some(VectorSize::Bi), None),
                    0b010 => (Some(VectorSize::Tri), None),
                    0b011 => (Some(VectorSize::Quad), None),
                    0b100 => (Some(VectorSize::Bi), Some(VectorSize::Bi)),
                    0b101 => (Some(VectorSize::Tri), Some(VectorSize::Tri)),
                    _ => (Some(VectorSize::Quad), Some(VectorSize::Quad)),
                };

                let args = vec![
                    match size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    },
                    match second_size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    },
                ];

                let fun = match name {
                    "max" => MacroCall::Splatted(MathFunction::Max, size, 1),
                    "min" => MacroCall::Splatted(MathFunction::Min, size, 1),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "mix" => {
            // bits layout
            // bit 0 trough 1 - dims
            // bit 2 trough 4 - types
            //
            // 0b10011 is the last element since splatted single elements
            // were already added
            for bits in 0..(0b10011) {
                let size = match bits & 0b11 {
                    0b00 => Some(VectorSize::Bi),
                    0b01 => Some(VectorSize::Tri),
                    0b10 => Some(VectorSize::Quad),
                    _ => None,
                };
                let (kind, splatted, boolean) = match bits >> 2 {
                    0b000 => (Sk::Sint, false, true),
                    0b001 => (Sk::Uint, false, true),
                    0b010 => (Sk::Float, false, true),
                    0b011 => (Sk::Float, false, false),
                    _ => (Sk::Float, true, false),
                };

                let ty = |kind, width| match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };
                let args = vec![
                    ty(kind, width),
                    ty(kind, width),
                    match (boolean, splatted) {
                        (true, _) => ty(Sk::Bool, crate::BOOL_WIDTH),
                        (_, false) => TypeInner::Scalar { kind, width },
                        _ => ty(kind, width),
                    },
                ];

                declaration.overloads.push(module.add_builtin(
                    args,
                    match boolean {
                        true => MacroCall::MixBoolean,
                        false => MacroCall::Splatted(MathFunction::Mix, size, 2),
                    },
                ))
            }
        }
        "clamp" => {
            // bits layout
            // bit 0 trough 1 - float/int/uint
            // bit 2 trough 3 - dims
            // bit 4 - splatted
            //
            // 0b11010 is the last element since splatted single elements
            // were already added
            for bits in 0..(0b11011) {
                let kind = match bits & 0b11 {
                    0b00 => Sk::Float,
                    0b01 => Sk::Sint,
                    0b10 => Sk::Uint,
                    _ => continue,
                };
                let size = match (bits >> 2) & 0b11 {
                    0b00 => Some(VectorSize::Bi),
                    0b01 => Some(VectorSize::Tri),
                    0b10 => Some(VectorSize::Quad),
                    _ => None,
                };
                let splatted = bits & 0b10000 == 0b10000;

                let base_ty = || match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };
                let limit_ty = || match splatted {
                    true => TypeInner::Scalar { kind, width },
                    false => base_ty(),
                };

                let args = vec![base_ty(), limit_ty(), limit_ty()];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::Clamp(size)))
            }
        }
        // Add common builtins with floats
        _ => inject_common_builtin(declaration, module, name, 4),
    }
}

/// Double version of [`inject_builtin`](inject_builtin)
pub fn inject_double_builtin(
    declaration: &mut FunctionDeclaration,
    module: &mut Module,
    name: &str,
) {
    declaration.double = true;
    let width = 8;
    match name {
        "abs" | "sign" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let kind = Sk::Float;

                let args = vec![match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                }];

                declaration.overloads.push(module.add_builtin(
                    args,
                    MacroCall::MathFunction(match name {
                        "abs" => MathFunction::Abs,
                        "sign" => MathFunction::Sign,
                        _ => unreachable!(),
                    }),
                ))
            }
        }
        "min" | "max" => {
            // bits layout
            // bit 0 trough 2 - dims
            for bits in 0..(0b111) {
                let (size, second_size) = match bits {
                    0b000 => (None, None),
                    0b001 => (Some(VectorSize::Bi), None),
                    0b010 => (Some(VectorSize::Tri), None),
                    0b011 => (Some(VectorSize::Quad), None),
                    0b100 => (Some(VectorSize::Bi), Some(VectorSize::Bi)),
                    0b101 => (Some(VectorSize::Tri), Some(VectorSize::Tri)),
                    _ => (Some(VectorSize::Quad), Some(VectorSize::Quad)),
                };
                let kind = Sk::Float;

                let args = vec![
                    match size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    },
                    match second_size {
                        Some(size) => TypeInner::Vector { size, kind, width },
                        None => TypeInner::Scalar { kind, width },
                    },
                ];

                let fun = match name {
                    "max" => MacroCall::Splatted(MathFunction::Max, size, 1),
                    "min" => MacroCall::Splatted(MathFunction::Min, size, 1),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "mix" => {
            // bits layout
            // bit 0 trough 1 - dims
            // bit 2 trough 3 - splatted/boolean
            //
            // 0b1010 is the last element since splatted with single elements
            // is equal to normal single elements
            for bits in 0..(0b1011) {
                let size = match bits & 0b11 {
                    0b00 => Some(VectorSize::Quad),
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => None,
                };
                let kind = Sk::Float;
                let (splatted, boolean) = match bits >> 2 {
                    0b00 => (false, false),
                    0b01 => (false, true),
                    _ => (true, false),
                };

                let ty = |kind, width| match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };
                let args = vec![
                    ty(kind, width),
                    ty(kind, width),
                    match (boolean, splatted) {
                        (true, _) => ty(Sk::Bool, crate::BOOL_WIDTH),
                        (_, false) => TypeInner::Scalar { kind, width },
                        _ => ty(kind, width),
                    },
                ];

                declaration.overloads.push(module.add_builtin(
                    args,
                    match boolean {
                        true => MacroCall::MixBoolean,
                        false => MacroCall::Splatted(MathFunction::Mix, size, 2),
                    },
                ))
            }
        }
        "clamp" => {
            // bits layout
            // bit 0 trough 1 - dims
            // bit 2 - splatted
            //
            // 0b110 is the last element since splatted with single elements
            // is equal to normal single elements
            for bits in 0..(0b111) {
                let kind = Sk::Float;
                let size = match bits & 0b11 {
                    0b00 => Some(VectorSize::Bi),
                    0b01 => Some(VectorSize::Tri),
                    0b10 => Some(VectorSize::Quad),
                    _ => None,
                };
                let splatted = bits & 0b100 == 0b100;

                let base_ty = || match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };
                let limit_ty = || match splatted {
                    true => TypeInner::Scalar { kind, width },
                    false => base_ty(),
                };

                let args = vec![base_ty(), limit_ty(), limit_ty()];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::Clamp(size)))
            }
        }
        // Add common builtins with doubles
        _ => inject_common_builtin(declaration, module, name, 8),
    }
}

fn inject_common_builtin(
    declaration: &mut FunctionDeclaration,
    module: &mut Module,
    name: &str,
    float_width: crate::Bytes,
) {
    match name {
        "ceil" | "round" | "roundEven" | "floor" | "fract" | "trunc" | "sqrt" | "inversesqrt"
        | "normalize" | "length" | "isinf" | "isnan" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                let args = vec![match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                }];

                let fun = match name {
                    "ceil" => MacroCall::MathFunction(MathFunction::Ceil),
                    "round" | "roundEven" => MacroCall::MathFunction(MathFunction::Round),
                    "floor" => MacroCall::MathFunction(MathFunction::Floor),
                    "fract" => MacroCall::MathFunction(MathFunction::Fract),
                    "trunc" => MacroCall::MathFunction(MathFunction::Trunc),
                    "sqrt" => MacroCall::MathFunction(MathFunction::Sqrt),
                    "inversesqrt" => MacroCall::MathFunction(MathFunction::InverseSqrt),
                    "normalize" => MacroCall::MathFunction(MathFunction::Normalize),
                    "length" => MacroCall::MathFunction(MathFunction::Length),
                    "isinf" => MacroCall::Relational(RelationalFunction::IsInf),
                    "isnan" => MacroCall::Relational(RelationalFunction::IsNan),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "dot" | "reflect" | "distance" | "ldexp" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let ty = || match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                };

                let fun = match name {
                    "dot" => MacroCall::MathFunction(MathFunction::Dot),
                    "reflect" => MacroCall::MathFunction(MathFunction::Reflect),
                    "distance" => MacroCall::MathFunction(MathFunction::Distance),
                    "ldexp" => MacroCall::MathFunction(MathFunction::Ldexp),
                    _ => unreachable!(),
                };

                declaration
                    .overloads
                    .push(module.add_builtin(vec![ty(), ty()], fun))
            }
        }
        "transpose" => {
            // bits layout
            // bit 0 trough 3 - dims
            for bits in 0..(0b1001) {
                let (rows, columns) = match bits {
                    0b0000 => (VectorSize::Bi, VectorSize::Bi),
                    0b0001 => (VectorSize::Bi, VectorSize::Tri),
                    0b0010 => (VectorSize::Bi, VectorSize::Quad),
                    0b0011 => (VectorSize::Tri, VectorSize::Bi),
                    0b0100 => (VectorSize::Tri, VectorSize::Tri),
                    0b0101 => (VectorSize::Tri, VectorSize::Quad),
                    0b0110 => (VectorSize::Quad, VectorSize::Bi),
                    0b0111 => (VectorSize::Quad, VectorSize::Tri),
                    _ => (VectorSize::Quad, VectorSize::Quad),
                };

                declaration.overloads.push(module.add_builtin(
                    vec![TypeInner::Matrix {
                        columns,
                        rows,
                        width: float_width,
                    }],
                    MacroCall::MathFunction(MathFunction::Transpose),
                ))
            }
        }
        "inverse" | "determinant" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b11) {
                let (rows, columns) = match bits {
                    0b00 => (VectorSize::Bi, VectorSize::Bi),
                    0b01 => (VectorSize::Tri, VectorSize::Tri),
                    _ => (VectorSize::Quad, VectorSize::Quad),
                };

                let args = vec![TypeInner::Matrix {
                    columns,
                    rows,
                    width: float_width,
                }];

                declaration.overloads.push(module.add_builtin(
                    args,
                    MacroCall::MathFunction(match name {
                        "inverse" => MathFunction::Inverse,
                        "determinant" => MathFunction::Determinant,
                        _ => unreachable!(),
                    }),
                ))
            }
        }
        "mod" | "step" => {
            // bits layout
            // bit 0 trough 2 - dims
            for bits in 0..(0b111) {
                let (size, second_size) = match bits {
                    0b000 => (None, None),
                    0b001 => (Some(VectorSize::Bi), None),
                    0b010 => (Some(VectorSize::Tri), None),
                    0b011 => (Some(VectorSize::Quad), None),
                    0b100 => (Some(VectorSize::Bi), Some(VectorSize::Bi)),
                    0b101 => (Some(VectorSize::Tri), Some(VectorSize::Tri)),
                    _ => (Some(VectorSize::Quad), Some(VectorSize::Quad)),
                };

                let mut args = Vec::with_capacity(2);
                let step = name == "step";

                for i in 0..2 {
                    let maybe_size = match i == step as u32 {
                        true => size,
                        false => second_size,
                    };

                    args.push(match maybe_size {
                        Some(size) => TypeInner::Vector {
                            size,
                            kind: Sk::Float,
                            width: float_width,
                        },
                        None => TypeInner::Scalar {
                            kind: Sk::Float,
                            width: float_width,
                        },
                    })
                }

                let fun = match name {
                    "mod" => MacroCall::Mod(size),
                    "step" => MacroCall::Splatted(MathFunction::Step, size, 0),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "modf" | "frexp" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                let ty = module.types.fetch_or_append(
                    Type {
                        name: None,
                        inner: match size {
                            Some(size) => TypeInner::Vector {
                                size,
                                kind: Sk::Float,
                                width: float_width,
                            },
                            None => TypeInner::Scalar {
                                kind: Sk::Float,
                                width: float_width,
                            },
                        },
                    },
                    Span::Unknown,
                );

                let parameters = vec![ty, ty];

                let fun = match name {
                    "modf" => MacroCall::MathFunction(MathFunction::Modf),
                    "frexp" => MacroCall::MathFunction(MathFunction::Frexp),
                    _ => unreachable!(),
                };

                declaration.overloads.push(Overload {
                    parameters,
                    parameters_info: vec![
                        ParameterInfo {
                            qualifier: ParameterQualifier::In,
                            depth: false,
                        },
                        ParameterInfo {
                            qualifier: ParameterQualifier::Out,
                            depth: false,
                        },
                    ],
                    kind: FunctionKind::Macro(fun),
                    defined: false,
                    void: false,
                })
            }
        }
        "cross" => {
            let args = vec![
                TypeInner::Vector {
                    size: VectorSize::Tri,
                    kind: Sk::Float,
                    width: float_width,
                },
                TypeInner::Vector {
                    size: VectorSize::Tri,
                    kind: Sk::Float,
                    width: float_width,
                },
            ];

            declaration
                .overloads
                .push(module.add_builtin(args, MacroCall::MathFunction(MathFunction::Cross)))
        }
        "outerProduct" => {
            // bits layout
            // bit 0 trough 3 - dims
            for bits in 0..(0b1001) {
                let (size1, size2) = match bits {
                    0b0000 => (VectorSize::Bi, VectorSize::Bi),
                    0b0001 => (VectorSize::Bi, VectorSize::Tri),
                    0b0010 => (VectorSize::Bi, VectorSize::Quad),
                    0b0011 => (VectorSize::Tri, VectorSize::Bi),
                    0b0100 => (VectorSize::Tri, VectorSize::Tri),
                    0b0101 => (VectorSize::Tri, VectorSize::Quad),
                    0b0110 => (VectorSize::Quad, VectorSize::Bi),
                    0b0111 => (VectorSize::Quad, VectorSize::Tri),
                    _ => (VectorSize::Quad, VectorSize::Quad),
                };

                let args = vec![
                    TypeInner::Vector {
                        size: size1,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    TypeInner::Vector {
                        size: size2,
                        kind: Sk::Float,
                        width: float_width,
                    },
                ];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::MathFunction(MathFunction::Outer)))
            }
        }
        "faceforward" | "fma" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                let ty = || match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                };
                let args = vec![ty(), ty(), ty()];

                let fun = match name {
                    "faceforward" => MacroCall::MathFunction(MathFunction::FaceForward),
                    "fma" => MacroCall::MathFunction(MathFunction::Fma),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "refract" => {
            // bits layout
            // bit 0 trough 1 - dims
            for bits in 0..(0b100) {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                let ty = || match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                };
                let args = vec![
                    ty(),
                    ty(),
                    TypeInner::Scalar {
                        kind: Sk::Float,
                        width: 4,
                    },
                ];
                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::MathFunction(MathFunction::Refract)))
            }
        }
        "smoothstep" => {
            // bits layout
            // bit 0 - splatted
            // bit 1 trough 2 - dims
            for bits in 0..(0b1000) {
                let splatted = bits & 0b1 == 0b1;
                let size = match bits >> 1 {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };

                if splatted && size.is_none() {
                    continue;
                }

                let base_ty = || match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind: Sk::Float,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                };
                let ty = || match splatted {
                    true => TypeInner::Scalar {
                        kind: Sk::Float,
                        width: float_width,
                    },
                    false => base_ty(),
                };
                declaration.overloads.push(module.add_builtin(
                    vec![ty(), ty(), base_ty()],
                    MacroCall::MathFunction(MathFunction::SmoothStep),
                ))
            }
        }
        // The function isn't a builtin or we don't yet support it
        _ => declaration.builtin = false,
    }
}

/// A compiler defined builtin function
#[derive(Clone, Copy, PartialEq)]
pub enum MacroCall {
    Sampler,
    SamplerShadow,
    Texture,
    TextureLod,
    TextureProj,
    TextureGrad(bool),
    TextureSize,
    TexelFetch,
    MathFunction(MathFunction),
    Relational(RelationalFunction),
    Binary(BinaryOperator),
    Mod(Option<VectorSize>),
    Splatted(MathFunction, Option<VectorSize>, usize),
    MixBoolean,
    Clamp(Option<VectorSize>),
    ConstMultiply(f64),
    BitCast(Sk),
    Derivate(DerivativeAxis),
}

impl MacroCall {
    /// Adds the necessary expressions and statements to the passed body and
    /// finally returns the final expression with the correct result
    pub fn call(
        &self,
        parser: &mut Parser,
        ctx: &mut Context,
        body: &mut Block,
        args: &mut [Handle<Expression>],
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>> {
        match *self {
            MacroCall::Sampler => {
                ctx.samplers.insert(args[0], args[1]);
                Ok(args[0])
            }
            MacroCall::SamplerShadow => {
                sampled_to_depth(&mut parser.module, ctx, args[0], meta, &mut parser.errors)?;
                parser.invalidate_expression(ctx, args[0], meta)?;
                ctx.samplers.insert(args[0], args[1]);
                Ok(args[0])
            }
            MacroCall::Texture | MacroCall::TextureLod => {
                let comps = parser.coordinate_components(ctx, args[0], args[1], meta, body)?;
                let level = match *self {
                    MacroCall::Texture => args
                        .get(2)
                        .copied()
                        .map_or(SampleLevel::Auto, SampleLevel::Bias),
                    MacroCall::TextureLod => SampleLevel::Exact(args[2]),
                    _ => unreachable!(),
                };
                texture_call(ctx, args[0], level, comps, body, meta)
            }
            MacroCall::TextureGrad(shadow) => {
                let comps = parser.coordinate_components(ctx, args[0], args[1], meta, body)?;
                let level = match shadow {
                    true => {
                        log::debug!(
                            "Assuming gradients {:?} and {:?} are not greater than 1",
                            args[2],
                            args[3]
                        );
                        SampleLevel::Zero
                    }
                    false => SampleLevel::Gradient {
                        x: args[2],
                        y: args[3],
                    },
                };
                texture_call(ctx, args[0], level, comps, body, meta)
            }
            MacroCall::TextureProj => {
                let level = args
                    .get(2)
                    .copied()
                    .map_or(SampleLevel::Auto, SampleLevel::Bias);
                let size = match *parser.resolve_type(ctx, args[1], meta)? {
                    TypeInner::Vector { size, .. } => size,
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::SemanticError("Bad call to textureProj".into()),
                            meta,
                        })
                    }
                };
                let base = args[1];
                let mut right = ctx.add_expression(
                    Expression::AccessIndex {
                        base,
                        index: size as u32 - 1,
                    },
                    SourceMetadata::none(),
                    body,
                );
                let left = if let VectorSize::Bi = size {
                    ctx.add_expression(
                        Expression::AccessIndex { base, index: 0 },
                        SourceMetadata::none(),
                        body,
                    )
                } else {
                    let size = match size {
                        VectorSize::Tri => VectorSize::Bi,
                        _ => VectorSize::Tri,
                    };
                    right = ctx.add_expression(
                        Expression::Splat { size, value: right },
                        SourceMetadata::none(),
                        body,
                    );
                    ctx.vector_resize(size, base, SourceMetadata::none(), body)
                };
                let coords = ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left,
                        right,
                    },
                    SourceMetadata::none(),
                    body,
                );
                let comps = parser.coordinate_components(ctx, args[0], coords, meta, body)?;
                texture_call(ctx, args[0], level, comps, body, meta)
            }
            MacroCall::TextureSize => Ok(ctx.add_expression(
                Expression::ImageQuery {
                    image: args[0],
                    query: ImageQuery::Size {
                        level: args.get(1).copied(),
                    },
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::TexelFetch => {
                let comps = parser.coordinate_components(ctx, args[0], args[1], meta, body)?;
                Ok(ctx.add_expression(
                    Expression::ImageLoad {
                        image: args[0],
                        coordinate: comps.coordinate,
                        array_index: comps.array_index,
                        index: Some(args[2]),
                    },
                    SourceMetadata::none(),
                    body,
                ))
            }
            MacroCall::MathFunction(fun) => Ok(ctx.add_expression(
                Expression::Math {
                    fun,
                    arg: args[0],
                    arg1: args.get(1).copied(),
                    arg2: args.get(2).copied(),
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::Relational(fun) => Ok(ctx.add_expression(
                Expression::Relational {
                    fun,
                    argument: args[0],
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::Binary(op) => Ok(ctx.add_expression(
                Expression::Binary {
                    op,
                    left: args[0],
                    right: args[1],
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::Mod(size) => {
                ctx.implicit_splat(parser, &mut args[1], meta, size)?;

                Ok(ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: args[0],
                        right: args[1],
                    },
                    SourceMetadata::none(),
                    body,
                ))
            }
            MacroCall::Splatted(fun, size, i) => {
                ctx.implicit_splat(parser, &mut args[i], meta, size)?;

                Ok(ctx.add_expression(
                    Expression::Math {
                        fun,
                        arg: args[0],
                        arg1: args.get(1).copied(),
                        arg2: args.get(2).copied(),
                    },
                    SourceMetadata::none(),
                    body,
                ))
            }
            MacroCall::MixBoolean => Ok(ctx.add_expression(
                Expression::Select {
                    condition: args[2],
                    accept: args[1],
                    reject: args[0],
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::Clamp(size) => {
                ctx.implicit_splat(parser, &mut args[1], meta, size)?;
                ctx.implicit_splat(parser, &mut args[2], meta, size)?;

                Ok(ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: args[0],
                        arg1: args.get(1).copied(),
                        arg2: args.get(2).copied(),
                    },
                    SourceMetadata::none(),
                    body,
                ))
            }
            MacroCall::ConstMultiply(value) => {
                let constant = parser.module.constants.fetch_or_append(
                    Constant {
                        name: None,
                        specialization: None,
                        inner: ConstantInner::Scalar {
                            width: 4,
                            value: ScalarValue::Float(value),
                        },
                    },
                    Span::Unknown,
                );
                let right = ctx.add_expression(
                    Expression::Constant(constant),
                    SourceMetadata::none(),
                    body,
                );
                Ok(ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: args[0],
                        right,
                    },
                    SourceMetadata::none(),
                    body,
                ))
            }
            MacroCall::BitCast(kind) => Ok(ctx.add_expression(
                Expression::As {
                    expr: args[0],
                    kind,
                    convert: None,
                },
                SourceMetadata::none(),
                body,
            )),
            MacroCall::Derivate(axis) => Ok(ctx.add_expression(
                Expression::Derivative {
                    axis,
                    expr: args[0],
                },
                SourceMetadata::none(),
                body,
            )),
        }
    }
}

fn texture_call(
    ctx: &mut Context,
    image: Handle<Expression>,
    level: SampleLevel,
    comps: CoordComponents,
    body: &mut Block,
    meta: SourceMetadata,
) -> Result<Handle<Expression>> {
    if let Some(sampler) = ctx.samplers.get(&image).copied() {
        Ok(ctx.add_expression(
            Expression::ImageSample {
                image,
                sampler,
                coordinate: comps.coordinate,
                array_index: comps.array_index,
                offset: None,
                level,
                depth_ref: comps.depth_ref,
            },
            meta,
            body,
        ))
    } else {
        Err(Error {
            kind: ErrorKind::SemanticError("Bad call".into()),
            meta,
        })
    }
}

/// Helper struct for texture calls with the separate components from the vector argument
///
/// Obtained by calling [`coordinate_components`](Parser::coordinate_components)
struct CoordComponents {
    coordinate: Handle<Expression>,
    depth_ref: Option<Handle<Expression>>,
    array_index: Option<Handle<Expression>>,
}

impl Parser {
    /// Helper function for texture calls, splits the vector argument into it's components
    fn coordinate_components(
        &mut self,
        ctx: &mut Context,
        image: Handle<Expression>,
        coord: Handle<Expression>,
        meta: SourceMetadata,
        body: &mut Block,
    ) -> Result<CoordComponents> {
        if let TypeInner::Image {
            dim,
            arrayed,
            class,
        } = *self.resolve_type(ctx, image, meta)?
        {
            let image_size = match dim {
                ImageDimension::D1 => None,
                ImageDimension::D2 => Some(VectorSize::Bi),
                ImageDimension::D3 => Some(VectorSize::Tri),
                ImageDimension::Cube => Some(VectorSize::Tri),
            };
            let coord_size = match *self.resolve_type(ctx, coord, meta)? {
                TypeInner::Vector { size, .. } => Some(size),
                _ => None,
            };
            let shadow = match class {
                ImageClass::Depth { .. } => true,
                _ => false,
            };

            let coordinate = match (image_size, coord_size) {
                (Some(size), Some(coord_s)) if size != coord_s => {
                    ctx.vector_resize(size, coord, SourceMetadata::none(), body)
                }
                (None, Some(_)) => ctx.add_expression(
                    Expression::AccessIndex {
                        base: coord,
                        index: 0,
                    },
                    SourceMetadata::none(),
                    body,
                ),
                _ => coord,
            };
            let array_index = match arrayed {
                true => {
                    let index = match shadow {
                        true => image_size.map_or(0, |s| s as u32 - 1),
                        false => image_size.map_or(0, |s| s as u32),
                    };

                    Some(ctx.add_expression(
                        Expression::AccessIndex { base: coord, index },
                        SourceMetadata::none(),
                        body,
                    ))
                }
                _ => None,
            };
            let depth_ref = match shadow {
                true => {
                    let index = image_size.map_or(0, |s| s as u32);

                    Some(ctx.add_expression(
                        Expression::AccessIndex { base: coord, index },
                        SourceMetadata::none(),
                        body,
                    ))
                }
                false => None,
            };

            Ok(CoordComponents {
                coordinate,
                depth_ref,
                array_index,
            })
        } else {
            self.errors.push(Error {
                kind: ErrorKind::SemanticError("Type is not an image".into()),
                meta,
            });

            Ok(CoordComponents {
                coordinate: coord,
                depth_ref: None,
                array_index: None,
            })
        }
    }
}

/// Helper function to cast a expression holding a sampled image to a
/// depth image.
pub fn sampled_to_depth(
    module: &mut Module,
    ctx: &mut Context,
    image: Handle<Expression>,
    meta: SourceMetadata,
    errors: &mut Vec<Error>,
) -> Result<()> {
    let ty = match ctx[image] {
        Expression::GlobalVariable(handle) => &mut module.global_variables.get_mut(handle).ty,
        Expression::FunctionArgument(i) => {
            ctx.parameters_info[i as usize].depth = true;
            &mut ctx.arguments[i as usize].ty
        }
        _ => {
            return Err(Error {
                kind: ErrorKind::SemanticError("Not a valid texture expression".into()),
                meta,
            })
        }
    };
    match module.types[*ty].inner {
        TypeInner::Image {
            class,
            dim,
            arrayed,
        } => match class {
            ImageClass::Sampled { multi, .. } => {
                *ty = module.types.fetch_or_append(
                    Type {
                        name: None,
                        inner: TypeInner::Image {
                            dim,
                            arrayed,
                            class: ImageClass::Depth { multi },
                        },
                    },
                    Span::Unknown,
                )
            }
            ImageClass::Depth { .. } => {}
            _ => errors.push(Error {
                kind: ErrorKind::SemanticError("Not a texture".into()),
                meta,
            }),
        },
        _ => errors.push(Error {
            kind: ErrorKind::SemanticError("Not a texture".into()),
            meta,
        }),
    };

    Ok(())
}
