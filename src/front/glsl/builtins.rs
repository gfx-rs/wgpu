use super::{
    ast::{FunctionDeclaration, FunctionKind, Overload, ParameterInfo, ParameterQualifier},
    context::Context,
    Error, ErrorKind, Parser, Result,
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
            parameters.push(self.types.insert(
                Type {
                    name: None,
                    inner: arg,
                },
                Span::default(),
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

fn make_coords_arg(number_of_components: usize, kind: Sk) -> TypeInner {
    let width = 4;

    match number_of_components {
        1 => TypeInner::Scalar { kind, width },
        _ => TypeInner::Vector {
            size: match number_of_components {
                2 => VectorSize::Bi,
                3 => VectorSize::Tri,
                _ => VectorSize::Quad,
            },
            kind,
            width,
        },
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
        "texture"
        | "textureGrad"
        | "textureGradOffset"
        | "textureLod"
        | "textureLodOffset"
        | "textureOffset"
        | "textureProj"
        | "textureProjGrad"
        | "textureProjGradOffset"
        | "textureProjLod"
        | "textureProjLodOffset"
        | "textureProjOffset" => {
            // bits layout
            // bits 0 through 1 - dims
            // bit 2 - shadow
            // bit 3 - array
            // bit 4 - extra variant
            // bit 5 - bias

            for bits in 0..(0b1000000) {
                let dim = bits & 0b11;
                let shadow = bits & 0b100 == 0b100;
                let arrayed = bits & 0b1000 == 0b1000;
                let variant = bits & 0b10000 == 0b10000;
                let bias = bits & 0b100000 == 0b100000;

                let builtin = match name {
                    // texture(gsampler, gvec P, [float bias]);
                    "texture" => MacroCall::Texture {
                        proj: false,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::None,
                    },
                    // textureGrad(gsampler, gvec P, gvec dPdx, gvec dPdy);
                    "textureGrad" => MacroCall::Texture {
                        proj: false,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::Grad,
                    },
                    // textureGradOffset(gsampler, gvec P, gvec dPdx, gvec dPdy, ivec offset);
                    "textureGradOffset" => MacroCall::Texture {
                        proj: false,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::Grad,
                    },
                    // textureLod(gsampler, gvec P, float lod);
                    "textureLod" => MacroCall::Texture {
                        proj: false,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::Lod,
                    },
                    // textureLodOffset(gsampler, gvec P, float lod, ivec offset);
                    "textureLodOffset" => MacroCall::Texture {
                        proj: false,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::Lod,
                    },
                    // textureOffset(gsampler, gvec+1 P, ivec offset, [float bias]);
                    "textureOffset" => MacroCall::Texture {
                        proj: false,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::None,
                    },
                    // textureProj(gsampler, gvec+1 P, [float bias]);
                    "textureProj" => MacroCall::Texture {
                        proj: true,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::None,
                    },
                    // textureProjGrad(gsampler, gvec+1 P, gvec dPdx, gvec dPdy);
                    "textureProjGrad" => MacroCall::Texture {
                        proj: true,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::Grad,
                    },
                    // textureProjGradOffset(gsampler, gvec+1 P, gvec dPdx, gvec dPdy, ivec offset);
                    "textureProjGradOffset" => MacroCall::Texture {
                        proj: true,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::Grad,
                    },
                    // textureProjLod(gsampler, gvec+1 P, float lod);
                    "textureProjLod" => MacroCall::Texture {
                        proj: true,
                        offset: false,
                        shadow,
                        level_type: TextureLevelType::Lod,
                    },
                    // textureProjLodOffset(gsampler, gvec+1 P, gvec dPdx, gvec dPdy, ivec offset);
                    "textureProjLodOffset" => MacroCall::Texture {
                        proj: true,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::Lod,
                    },
                    // textureProjOffset(gsampler, gvec+1 P, ivec offset, [float bias]);
                    "textureProjOffset" => MacroCall::Texture {
                        proj: true,
                        offset: true,
                        shadow,
                        level_type: TextureLevelType::None,
                    },
                    _ => unreachable!(),
                };

                // Parse out the variant settings.
                let proj = matches!(builtin, MacroCall::Texture { proj: true, .. });
                let grad = matches!(
                    builtin,
                    MacroCall::Texture {
                        level_type: TextureLevelType::Grad,
                        ..
                    }
                );
                let lod = matches!(
                    builtin,
                    MacroCall::Texture {
                        level_type: TextureLevelType::Lod,
                        ..
                    }
                );
                let offset = matches!(builtin, MacroCall::Texture { offset: true, .. });

                let supports_variant = proj && !shadow;
                if variant && !supports_variant {
                    continue;
                }

                let supports_bias = matches!(
                    builtin,
                    MacroCall::Texture {
                        level_type: TextureLevelType::None,
                        ..
                    }
                ) && !shadow;
                if bias && !supports_bias {
                    continue;
                }

                // Proj doesn't work with arrayed, Cube or 3D samplers
                if proj && (arrayed || dim == 0b10 || dim == 0b11) {
                    continue;
                }

                // 3DArray and 3DShadow are not valid texture types
                if dim == 0b11 && (arrayed || shadow) {
                    continue;
                }

                // It seems that textureGradOffset(samplerCube) is not defined by GLSL for some reason...
                if dim == 0b10 && grad && offset {
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

                let num_coords_from_dim = (dim + 1).min(3);
                let mut num_coords = num_coords_from_dim;

                if shadow && proj {
                    num_coords = 4;
                } else if shadow {
                    num_coords += 1;
                } else if proj {
                    if variant && num_coords == 4 {
                        // Normal form already has 4 components, no need to have a variant form.
                        continue;
                    } else if variant {
                        num_coords = 4;
                    } else {
                        num_coords += 1;
                    }
                }

                num_coords += arrayed as usize;

                // Special case: texture(gsamplerCubeArrayShadow) kicks the shadow compare ref to a separate argument,
                // since it would otherwise take five arguments. It also can't take a bias, nor can it be proj/grad/lod/offset
                // (presumably because nobody asked for it, and implementation complexity?)
                if num_coords >= 5 {
                    if lod || grad || offset || proj || bias {
                        continue;
                    }
                    debug_assert!(dim == 0b10 && shadow && arrayed);
                }
                debug_assert!(num_coords <= 5);

                let vector = make_coords_arg(num_coords, Sk::Float);
                let mut args = vec![image, vector];

                if num_coords == 5 {
                    args.push(TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    });
                }

                match builtin {
                    MacroCall::Texture {
                        level_type: TextureLevelType::Lod,
                        ..
                    } => {
                        args.push(TypeInner::Scalar {
                            kind: Sk::Float,
                            width,
                        });
                    }
                    MacroCall::Texture {
                        level_type: TextureLevelType::Grad,
                        ..
                    } => {
                        args.push(make_coords_arg(num_coords_from_dim, Sk::Float));
                        args.push(make_coords_arg(num_coords_from_dim, Sk::Float));
                    }
                    _ => {}
                };

                if offset {
                    args.push(make_coords_arg(num_coords_from_dim, Sk::Sint));
                }

                if bias {
                    args.push(TypeInner::Scalar {
                        kind: Sk::Float,
                        width,
                    });
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, builtin));
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
                let arrayed = bits & 0b10 == 0b10;

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
        "bitCount" | "bitfieldReverse" | "bitfieldExtract" | "bitfieldInsert" => {
            let fun = match name {
                "bitCount" => MathFunction::CountOneBits,
                "bitfieldReverse" => MathFunction::ReverseBits,
                "bitfieldExtract" => MathFunction::ExtractBits,
                "bitfieldInsert" => MathFunction::InsertBits,
                _ => unreachable!(),
            };

            let mc = match fun {
                MathFunction::ExtractBits => MacroCall::BitfieldExtract,
                MathFunction::InsertBits => MacroCall::BitfieldInsert,
                _ => MacroCall::MathFunction(fun),
            };

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

                let ty = || match size {
                    Some(size) => TypeInner::Vector { size, kind, width },
                    None => TypeInner::Scalar { kind, width },
                };

                let mut args = vec![ty()];

                match fun {
                    MathFunction::ExtractBits => {
                        args.push(TypeInner::Scalar {
                            kind: Sk::Sint,
                            width: 4,
                        });
                        args.push(TypeInner::Scalar {
                            kind: Sk::Sint,
                            width: 4,
                        });
                    }
                    MathFunction::InsertBits => {
                        args.push(ty());
                        args.push(TypeInner::Scalar {
                            kind: Sk::Sint,
                            width: 4,
                        });
                        args.push(TypeInner::Scalar {
                            kind: Sk::Sint,
                            width: 4,
                        });
                    }
                    _ => {}
                }

                declaration.overloads.push(module.add_builtin(args, mc))
            }
        }
        "packSnorm4x8" | "packUnorm4x8" | "packSnorm2x16" | "packUnorm2x16" | "packHalf2x16" => {
            let fun = match name {
                "packSnorm4x8" => MathFunction::Pack4x8snorm,
                "packUnorm4x8" => MathFunction::Pack4x8unorm,
                "packSnorm2x16" => MathFunction::Pack2x16unorm,
                "packUnorm2x16" => MathFunction::Pack2x16snorm,
                "packHalf2x16" => MathFunction::Pack2x16float,
                _ => unreachable!(),
            };

            let ty = match fun {
                MathFunction::Pack4x8snorm | MathFunction::Pack4x8unorm => TypeInner::Vector {
                    size: crate::VectorSize::Quad,
                    kind: Sk::Float,
                    width: 4,
                },
                MathFunction::Pack2x16unorm
                | MathFunction::Pack2x16snorm
                | MathFunction::Pack2x16float => TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    kind: Sk::Float,
                    width: 4,
                },
                _ => unreachable!(),
            };

            let args = vec![ty];

            declaration
                .overloads
                .push(module.add_builtin(args, MacroCall::MathFunction(fun)));
        }
        "unpackSnorm4x8" | "unpackUnorm4x8" | "unpackSnorm2x16" | "unpackUnorm2x16"
        | "unpackHalf2x16" => {
            let fun = match name {
                "unpackSnorm4x8" => MathFunction::Unpack4x8snorm,
                "unpackUnorm4x8" => MathFunction::Unpack4x8unorm,
                "unpackSnorm2x16" => MathFunction::Unpack2x16snorm,
                "unpackUnorm2x16" => MathFunction::Unpack2x16unorm,
                "unpackHalf2x16" => MathFunction::Unpack2x16float,
                _ => unreachable!(),
            };

            let args = vec![TypeInner::Scalar {
                kind: Sk::Uint,
                width: 4,
            }];

            declaration
                .overloads
                .push(module.add_builtin(args, MacroCall::MathFunction(fun)));
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

                let ty = module.types.insert(
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
                    Span::default(),
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TextureLevelType {
    None,
    Lod,
    Grad,
}

/// A compiler defined builtin function
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MacroCall {
    Sampler,
    SamplerShadow,
    Texture {
        proj: bool,
        offset: bool,
        shadow: bool,
        level_type: TextureLevelType,
    },
    TextureSize,
    TexelFetch,
    MathFunction(MathFunction),
    BitfieldExtract,
    BitfieldInsert,
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
        meta: Span,
    ) -> Result<Handle<Expression>> {
        match *self {
            MacroCall::Sampler => {
                ctx.samplers.insert(args[0], args[1]);
                Ok(args[0])
            }
            MacroCall::SamplerShadow => {
                sampled_to_depth(&mut parser.module, ctx, args[0], meta, &mut parser.errors);
                parser.invalidate_expression(ctx, args[0], meta)?;
                ctx.samplers.insert(args[0], args[1]);
                Ok(args[0])
            }
            MacroCall::Texture {
                proj,
                offset,
                shadow,
                level_type,
            } => {
                let mut coords = args[1];

                if proj {
                    let size = match *parser.resolve_type(ctx, coords, meta)? {
                        TypeInner::Vector { size, .. } => size,
                        _ => unreachable!(),
                    };
                    let mut right = ctx.add_expression(
                        Expression::AccessIndex {
                            base: coords,
                            index: size as u32 - 1,
                        },
                        Span::default(),
                        body,
                    );
                    let left = if let VectorSize::Bi = size {
                        ctx.add_expression(
                            Expression::AccessIndex {
                                base: coords,
                                index: 0,
                            },
                            Span::default(),
                            body,
                        )
                    } else {
                        let size = match size {
                            VectorSize::Tri => VectorSize::Bi,
                            _ => VectorSize::Tri,
                        };
                        right = ctx.add_expression(
                            Expression::Splat { size, value: right },
                            Span::default(),
                            body,
                        );
                        ctx.vector_resize(size, coords, Span::default(), body)
                    };
                    coords = ctx.add_expression(
                        Expression::Binary {
                            op: BinaryOperator::Divide,
                            left,
                            right,
                        },
                        Span::default(),
                        body,
                    );
                }

                let extra = args.get(2).copied();
                let comps =
                    parser.coordinate_components(ctx, args[0], coords, extra, meta, body)?;

                let mut num_args = 2;

                if comps.used_extra {
                    num_args += 1;
                };

                // Parse out explicit texture level.
                let mut level = match level_type {
                    TextureLevelType::None => SampleLevel::Auto,

                    TextureLevelType::Lod => {
                        num_args += 1;

                        if shadow {
                            log::warn!("Assuming LOD {:?} is zero", args[2],);

                            SampleLevel::Zero
                        } else {
                            SampleLevel::Exact(args[2])
                        }
                    }

                    TextureLevelType::Grad => {
                        num_args += 2;

                        if shadow {
                            log::warn!(
                                "Assuming gradients {:?} and {:?} are not greater than 1",
                                args[2],
                                args[3],
                            );
                            SampleLevel::Zero
                        } else {
                            SampleLevel::Gradient {
                                x: args[2],
                                y: args[3],
                            }
                        }
                    }
                };

                let texture_offset = match offset {
                    true => {
                        let offset_arg = args[num_args];
                        num_args += 1;
                        match parser.solve_constant(ctx, offset_arg, meta) {
                            Ok(v) => Some(v),
                            Err(e) => {
                                parser.errors.push(e);
                                None
                            }
                        }
                    }
                    false => None,
                };

                // Now go back and look for optional bias arg (if available)
                if let TextureLevelType::None = level_type {
                    level = args
                        .get(num_args)
                        .copied()
                        .map_or(SampleLevel::Auto, SampleLevel::Bias);
                }

                texture_call(ctx, args[0], level, comps, texture_offset, body, meta)
            }
            MacroCall::TextureSize => Ok(ctx.add_expression(
                Expression::ImageQuery {
                    image: args[0],
                    query: ImageQuery::Size {
                        level: args.get(1).copied(),
                    },
                },
                Span::default(),
                body,
            )),
            MacroCall::TexelFetch => {
                let comps =
                    parser.coordinate_components(ctx, args[0], args[1], None, meta, body)?;
                Ok(ctx.add_expression(
                    Expression::ImageLoad {
                        image: args[0],
                        coordinate: comps.coordinate,
                        array_index: comps.array_index,
                        index: Some(args[2]),
                    },
                    Span::default(),
                    body,
                ))
            }
            MacroCall::MathFunction(fun) => Ok(ctx.add_expression(
                Expression::Math {
                    fun,
                    arg: args[0],
                    arg1: args.get(1).copied(),
                    arg2: args.get(2).copied(),
                    arg3: args.get(3).copied(),
                },
                Span::default(),
                body,
            )),
            MacroCall::BitfieldInsert => {
                let conv_arg_2 = ctx.add_expression(
                    Expression::As {
                        expr: args[2],
                        kind: Sk::Uint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                );
                let conv_arg_3 = ctx.add_expression(
                    Expression::As {
                        expr: args[3],
                        kind: Sk::Uint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                );
                Ok(ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::InsertBits,
                        arg: args[0],
                        arg1: Some(args[1]),
                        arg2: Some(conv_arg_2),
                        arg3: Some(conv_arg_3),
                    },
                    Span::default(),
                    body,
                ))
            }
            MacroCall::BitfieldExtract => {
                let conv_arg_1 = ctx.add_expression(
                    Expression::As {
                        expr: args[1],
                        kind: Sk::Uint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                );
                let conv_arg_2 = ctx.add_expression(
                    Expression::As {
                        expr: args[2],
                        kind: Sk::Uint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                );
                Ok(ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::ExtractBits,
                        arg: args[0],
                        arg1: Some(conv_arg_1),
                        arg2: Some(conv_arg_2),
                        arg3: None,
                    },
                    Span::default(),
                    body,
                ))
            }
            MacroCall::Relational(fun) => Ok(ctx.add_expression(
                Expression::Relational {
                    fun,
                    argument: args[0],
                },
                Span::default(),
                body,
            )),
            MacroCall::Binary(op) => Ok(ctx.add_expression(
                Expression::Binary {
                    op,
                    left: args[0],
                    right: args[1],
                },
                Span::default(),
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
                    Span::default(),
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
                        arg3: args.get(3).copied(),
                    },
                    Span::default(),
                    body,
                ))
            }
            MacroCall::MixBoolean => Ok(ctx.add_expression(
                Expression::Select {
                    condition: args[2],
                    accept: args[1],
                    reject: args[0],
                },
                Span::default(),
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
                        arg3: args.get(3).copied(),
                    },
                    Span::default(),
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
                    Span::default(),
                );
                let right =
                    ctx.add_expression(Expression::Constant(constant), Span::default(), body);
                Ok(ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: args[0],
                        right,
                    },
                    Span::default(),
                    body,
                ))
            }
            MacroCall::BitCast(kind) => Ok(ctx.add_expression(
                Expression::As {
                    expr: args[0],
                    kind,
                    convert: None,
                },
                Span::default(),
                body,
            )),
            MacroCall::Derivate(axis) => Ok(ctx.add_expression(
                Expression::Derivative {
                    axis,
                    expr: args[0],
                },
                Span::default(),
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
    offset: Option<Handle<Constant>>,
    body: &mut Block,
    meta: Span,
) -> Result<Handle<Expression>> {
    if let Some(sampler) = ctx.samplers.get(&image).copied() {
        let mut array_index = comps.array_index;

        if let Some(ref mut array_index_expr) = array_index {
            ctx.conversion(array_index_expr, meta, Sk::Sint, 4)?;
        }

        Ok(ctx.add_expression(
            Expression::ImageSample {
                image,
                sampler,
                coordinate: comps.coordinate,
                array_index,
                offset,
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
#[derive(Debug)]
struct CoordComponents {
    coordinate: Handle<Expression>,
    depth_ref: Option<Handle<Expression>>,
    array_index: Option<Handle<Expression>>,
    used_extra: bool,
}

impl Parser {
    /// Helper function for texture calls, splits the vector argument into it's components
    fn coordinate_components(
        &mut self,
        ctx: &mut Context,
        image: Handle<Expression>,
        coord: Handle<Expression>,
        extra: Option<Handle<Expression>>,
        meta: Span,
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
                    ctx.vector_resize(size, coord, Span::default(), body)
                }
                (None, Some(_)) => ctx.add_expression(
                    Expression::AccessIndex {
                        base: coord,
                        index: 0,
                    },
                    Span::default(),
                    body,
                ),
                _ => coord,
            };

            let mut coord_index = image_size.map_or(1, |s| s as u32);

            let array_index = match arrayed {
                true => {
                    let index = coord_index;
                    coord_index += 1;

                    Some(ctx.add_expression(
                        Expression::AccessIndex { base: coord, index },
                        Span::default(),
                        body,
                    ))
                }
                _ => None,
            };
            let mut used_extra = false;
            let depth_ref = match shadow {
                true => {
                    let index = coord_index;

                    if index == 4 {
                        used_extra = true;
                        extra
                    } else {
                        Some(ctx.add_expression(
                            Expression::AccessIndex { base: coord, index },
                            Span::default(),
                            body,
                        ))
                    }
                }
                false => None,
            };

            Ok(CoordComponents {
                coordinate,
                depth_ref,
                array_index,
                used_extra,
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
                used_extra: false,
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
    meta: Span,
    errors: &mut Vec<Error>,
) {
    let ty = match ctx[image] {
        Expression::GlobalVariable(handle) => &mut module.global_variables.get_mut(handle).ty,
        Expression::FunctionArgument(i) => {
            ctx.parameters_info[i as usize].depth = true;
            &mut ctx.arguments[i as usize].ty
        }
        _ => {
            return errors.push(Error {
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
                *ty = module.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Image {
                            dim,
                            arrayed,
                            class: ImageClass::Depth { multi },
                        },
                    },
                    Span::default(),
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
}
