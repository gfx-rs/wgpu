use super::{
    ast::{
        BuiltinVariations, FunctionDeclaration, FunctionKind, Overload, ParameterInfo,
        ParameterQualifier,
    },
    context::Context,
    Error, ErrorKind, Frontend, Result,
};
use crate::{
    BinaryOperator, Block, DerivativeAxis as Axis, DerivativeControl as Ctrl, Expression, Handle,
    ImageClass, ImageDimension as Dim, ImageQuery, MathFunction, Module, RelationalFunction,
    SampleLevel, ScalarKind as Sk, Span, Type, TypeInner, UnaryOperator, VectorSize,
};

impl crate::ScalarKind {
    const fn dummy_storage_format(&self) -> crate::StorageFormat {
        match *self {
            Sk::Sint => crate::StorageFormat::R16Sint,
            Sk::Uint => crate::StorageFormat::R16Uint,
            _ => crate::StorageFormat::R16Float,
        }
    }
}

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
            internal: true,
            void: false,
        }
    }
}

const fn make_coords_arg(number_of_components: usize, kind: Sk) -> TypeInner {
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

/// Inject builtins into the declaration
///
/// This is done to not add a large startup cost and not increase memory
/// usage if it isn't needed.
pub fn inject_builtin(
    declaration: &mut FunctionDeclaration,
    module: &mut Module,
    name: &str,
    mut variations: BuiltinVariations,
) {
    log::trace!(
        "{} variations: {:?} {:?}",
        name,
        variations,
        declaration.variations
    );
    // Don't regeneate variations
    variations.remove(declaration.variations);
    declaration.variations |= variations;

    if variations.contains(BuiltinVariations::STANDARD) {
        inject_standard_builtins(declaration, module, name)
    }

    if variations.contains(BuiltinVariations::DOUBLE) {
        inject_double_builtin(declaration, module, name)
    }

    let width = 4;
    match name {
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
            let f = |kind, dim, arrayed, multi, shadow| {
                for bits in 0..=0b11 {
                    let variant = bits & 0b1 != 0;
                    let bias = bits & 0b10 != 0;

                    let (proj, offset, level_type) = match name {
                        // texture(gsampler, gvec P, [float bias]);
                        "texture" => (false, false, TextureLevelType::None),
                        // textureGrad(gsampler, gvec P, gvec dPdx, gvec dPdy);
                        "textureGrad" => (false, false, TextureLevelType::Grad),
                        // textureGradOffset(gsampler, gvec P, gvec dPdx, gvec dPdy, ivec offset);
                        "textureGradOffset" => (false, true, TextureLevelType::Grad),
                        // textureLod(gsampler, gvec P, float lod);
                        "textureLod" => (false, false, TextureLevelType::Lod),
                        // textureLodOffset(gsampler, gvec P, float lod, ivec offset);
                        "textureLodOffset" => (false, true, TextureLevelType::Lod),
                        // textureOffset(gsampler, gvec+1 P, ivec offset, [float bias]);
                        "textureOffset" => (false, true, TextureLevelType::None),
                        // textureProj(gsampler, gvec+1 P, [float bias]);
                        "textureProj" => (true, false, TextureLevelType::None),
                        // textureProjGrad(gsampler, gvec+1 P, gvec dPdx, gvec dPdy);
                        "textureProjGrad" => (true, false, TextureLevelType::Grad),
                        // textureProjGradOffset(gsampler, gvec+1 P, gvec dPdx, gvec dPdy, ivec offset);
                        "textureProjGradOffset" => (true, true, TextureLevelType::Grad),
                        // textureProjLod(gsampler, gvec+1 P, float lod);
                        "textureProjLod" => (true, false, TextureLevelType::Lod),
                        // textureProjLodOffset(gsampler, gvec+1 P, gvec dPdx, gvec dPdy, ivec offset);
                        "textureProjLodOffset" => (true, true, TextureLevelType::Lod),
                        // textureProjOffset(gsampler, gvec+1 P, ivec offset, [float bias]);
                        "textureProjOffset" => (true, true, TextureLevelType::None),
                        _ => unreachable!(),
                    };

                    let builtin = MacroCall::Texture {
                        proj,
                        offset,
                        shadow,
                        level_type,
                    };

                    // Parse out the variant settings.
                    let grad = level_type == TextureLevelType::Grad;
                    let lod = level_type == TextureLevelType::Lod;

                    let supports_variant = proj && !shadow;
                    if variant && !supports_variant {
                        continue;
                    }

                    if bias && !matches!(level_type, TextureLevelType::None) {
                        continue;
                    }

                    // Proj doesn't work with arrayed or Cube
                    if proj && (arrayed || dim == Dim::Cube) {
                        continue;
                    }

                    // texture operations with offset are not supported for cube maps
                    if dim == Dim::Cube && offset {
                        continue;
                    }

                    // sampler2DArrayShadow can't be used in textureLod or in texture with bias
                    if (lod || bias) && arrayed && shadow && dim == Dim::D2 {
                        continue;
                    }

                    // TODO: glsl supports using bias with depth samplers but naga doesn't
                    if bias && shadow {
                        continue;
                    }

                    let class = match shadow {
                        true => ImageClass::Depth { multi },
                        false => ImageClass::Sampled { kind, multi },
                    };

                    let image = TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    };

                    let num_coords_from_dim = image_dims_to_coords_size(dim).min(3);
                    let mut num_coords = num_coords_from_dim;

                    if shadow && proj {
                        num_coords = 4;
                    } else if dim == Dim::D1 && shadow {
                        num_coords = 3;
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

                    if !(dim == Dim::D1 && shadow) {
                        num_coords += arrayed as usize;
                    }

                    // Special case: texture(gsamplerCubeArrayShadow) kicks the shadow compare ref to a separate argument,
                    // since it would otherwise take five arguments. It also can't take a bias, nor can it be proj/grad/lod/offset
                    // (presumably because nobody asked for it, and implementation complexity?)
                    if num_coords >= 5 {
                        if lod || grad || offset || proj || bias {
                            continue;
                        }
                        debug_assert!(dim == Dim::Cube && shadow && arrayed);
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

                    match level_type {
                        TextureLevelType::Lod => {
                            args.push(TypeInner::Scalar {
                                kind: Sk::Float,
                                width,
                            });
                        }
                        TextureLevelType::Grad => {
                            args.push(make_coords_arg(num_coords_from_dim, Sk::Float));
                            args.push(make_coords_arg(num_coords_from_dim, Sk::Float));
                        }
                        TextureLevelType::None => {}
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
            };

            texture_args_generator(TextureArgsOptions::SHADOW | variations.into(), f)
        }
        "textureSize" => {
            let f = |kind, dim, arrayed, multi, shadow| {
                let class = match shadow {
                    true => ImageClass::Depth { multi },
                    false => ImageClass::Sampled { kind, multi },
                };

                let image = TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                };

                let mut args = vec![image];

                if !multi {
                    args.push(TypeInner::Scalar {
                        kind: Sk::Sint,
                        width,
                    })
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::TextureSize { arrayed }))
            };

            texture_args_generator(
                TextureArgsOptions::SHADOW | TextureArgsOptions::MULTI | variations.into(),
                f,
            )
        }
        "texelFetch" | "texelFetchOffset" => {
            let offset = "texelFetchOffset" == name;
            let f = |kind, dim, arrayed, multi, _shadow| {
                // Cube images aren't supported
                if let Dim::Cube = dim {
                    return;
                }

                let image = TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Sampled { kind, multi },
                };

                let dim_value = image_dims_to_coords_size(dim);
                let coordinates = make_coords_arg(dim_value + arrayed as usize, Sk::Sint);

                let mut args = vec![
                    image,
                    coordinates,
                    TypeInner::Scalar {
                        kind: Sk::Sint,
                        width,
                    },
                ];

                if offset {
                    args.push(make_coords_arg(dim_value, Sk::Sint));
                }

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::ImageLoad { multi }))
            };

            // Don't generate shadow images since they aren't supported
            texture_args_generator(TextureArgsOptions::MULTI | variations.into(), f)
        }
        "imageSize" => {
            let f = |kind: Sk, dim, arrayed, _, _| {
                // Naga doesn't support cube images and it's usefulness
                // is questionable, so they won't be supported for now
                if dim == Dim::Cube {
                    return;
                }

                let image = TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Storage {
                        format: kind.dummy_storage_format(),
                        access: crate::StorageAccess::empty(),
                    },
                };

                declaration
                    .overloads
                    .push(module.add_builtin(vec![image], MacroCall::TextureSize { arrayed }))
            };

            texture_args_generator(variations.into(), f)
        }
        "imageLoad" => {
            let f = |kind: Sk, dim, arrayed, _, _| {
                // Naga doesn't support cube images and it's usefulness
                // is questionable, so they won't be supported for now
                if dim == Dim::Cube {
                    return;
                }

                let image = TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Storage {
                        format: kind.dummy_storage_format(),
                        access: crate::StorageAccess::LOAD,
                    },
                };

                let dim_value = image_dims_to_coords_size(dim);
                let mut coord_size = dim_value + arrayed as usize;
                // > Every OpenGL API call that operates on cubemap array
                // > textures takes layer-faces, not array layers
                //
                // So this means that imageCubeArray only takes a three component
                // vector coordinate and the third component is a layer index.
                if Dim::Cube == dim && arrayed {
                    coord_size = 3
                }
                let coordinates = make_coords_arg(coord_size, Sk::Sint);

                let args = vec![image, coordinates];

                declaration
                    .overloads
                    .push(module.add_builtin(args, MacroCall::ImageLoad { multi: false }))
            };

            // Don't generate shadow nor multisampled images since they aren't supported
            texture_args_generator(variations.into(), f)
        }
        "imageStore" => {
            let f = |kind: Sk, dim, arrayed, _, _| {
                // Naga doesn't support cube images and it's usefulness
                // is questionable, so they won't be supported for now
                if dim == Dim::Cube {
                    return;
                }

                let image = TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Storage {
                        format: kind.dummy_storage_format(),
                        access: crate::StorageAccess::STORE,
                    },
                };

                let dim_value = image_dims_to_coords_size(dim);
                let mut coord_size = dim_value + arrayed as usize;
                // > Every OpenGL API call that operates on cubemap array
                // > textures takes layer-faces, not array layers
                //
                // So this means that imageCubeArray only takes a three component
                // vector coordinate and the third component is a layer index.
                if Dim::Cube == dim && arrayed {
                    coord_size = 3
                }
                let coordinates = make_coords_arg(coord_size, Sk::Sint);

                let args = vec![
                    image,
                    coordinates,
                    TypeInner::Vector {
                        size: VectorSize::Quad,
                        kind,
                        width,
                    },
                ];

                let mut overload = module.add_builtin(args, MacroCall::ImageStore);
                overload.void = true;
                declaration.overloads.push(overload)
            };

            // Don't generate shadow nor multisampled images since they aren't supported
            texture_args_generator(variations.into(), f)
        }
        _ => {}
    }
}

/// Injects the builtins into declaration that don't need any special variations
fn inject_standard_builtins(
    declaration: &mut FunctionDeclaration,
    module: &mut Module,
    name: &str,
) {
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
        "sin" | "exp" | "exp2" | "sinh" | "cos" | "cosh" | "tan" | "tanh" | "acos" | "asin"
        | "log" | "log2" | "radians" | "degrees" | "asinh" | "acosh" | "atanh"
        | "floatBitsToInt" | "floatBitsToUint" | "dFdx" | "dFdxFine" | "dFdxCoarse" | "dFdy"
        | "dFdyFine" | "dFdyCoarse" | "fwidth" | "fwidthFine" | "fwidthCoarse" => {
            // bits layout
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
                        "radians" => MacroCall::MathFunction(MathFunction::Radians),
                        "degrees" => MacroCall::MathFunction(MathFunction::Degrees),
                        "floatBitsToInt" => MacroCall::BitCast(Sk::Sint),
                        "floatBitsToUint" => MacroCall::BitCast(Sk::Uint),
                        "dFdxCoarse" => MacroCall::Derivate(Axis::X, Ctrl::Coarse),
                        "dFdyCoarse" => MacroCall::Derivate(Axis::Y, Ctrl::Coarse),
                        "fwidthCoarse" => MacroCall::Derivate(Axis::Width, Ctrl::Coarse),
                        "dFdxFine" => MacroCall::Derivate(Axis::X, Ctrl::Fine),
                        "dFdyFine" => MacroCall::Derivate(Axis::Y, Ctrl::Fine),
                        "fwidthFine" => MacroCall::Derivate(Axis::Width, Ctrl::Fine),
                        "dFdx" => MacroCall::Derivate(Axis::X, Ctrl::None),
                        "dFdy" => MacroCall::Derivate(Axis::Y, Ctrl::None),
                        "fwidth" => MacroCall::Derivate(Axis::Width, Ctrl::None),
                        _ => unreachable!(),
                    },
                ))
            }
        }
        "intBitsToFloat" | "uintBitsToFloat" => {
            // bits layout
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 0 through 1 - dims
            // bit 2 - float/sint
            for bits in 0..0b1000 {
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
        "bitCount" | "bitfieldReverse" | "bitfieldExtract" | "bitfieldInsert" | "findLSB"
        | "findMSB" => {
            let fun = match name {
                "bitCount" => MathFunction::CountOneBits,
                "bitfieldReverse" => MathFunction::ReverseBits,
                "bitfieldExtract" => MathFunction::ExtractBits,
                "bitfieldInsert" => MathFunction::InsertBits,
                "findLSB" => MathFunction::FindLsb,
                "findMSB" => MathFunction::FindMsb,
                _ => unreachable!(),
            };

            let mc = match fun {
                MathFunction::ExtractBits => MacroCall::BitfieldExtract,
                MathFunction::InsertBits => MacroCall::BitfieldInsert,
                _ => MacroCall::MathFunction(fun),
            };

            // bits layout
            // bit 0 - int/uint
            // bit 1 through 2 - dims
            for bits in 0..0b1000 {
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

                // we need to cast the return type of findLsb / findMsb
                let mc = if kind == Sk::Uint {
                    match mc {
                        MacroCall::MathFunction(MathFunction::FindLsb) => MacroCall::FindLsbUint,
                        MacroCall::MathFunction(MathFunction::FindMsb) => MacroCall::FindMsbUint,
                        mc => mc,
                    }
                } else {
                    mc
                };

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
            // bit 1 through 2 - dims
            for bits in 0..0b1000 {
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
        "all" | "any" | "not" => {
            // bits layout
            // bit 0 through 1 - dims
            for bits in 0..0b11 {
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

                let fun = match name {
                    "all" => MacroCall::Relational(RelationalFunction::All),
                    "any" => MacroCall::Relational(RelationalFunction::Any),
                    "not" => MacroCall::Unary(UnaryOperator::Not),
                    _ => unreachable!(),
                };

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" => {
            for bits in 0..0b1001 {
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
                    _ => unreachable!(),
                });

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "equal" | "notEqual" => {
            for bits in 0..0b1100 {
                let (size, kind) = match bits {
                    0b0000 => (VectorSize::Bi, Sk::Float),
                    0b0001 => (VectorSize::Tri, Sk::Float),
                    0b0010 => (VectorSize::Quad, Sk::Float),
                    0b0011 => (VectorSize::Bi, Sk::Sint),
                    0b0100 => (VectorSize::Tri, Sk::Sint),
                    0b0101 => (VectorSize::Quad, Sk::Sint),
                    0b0110 => (VectorSize::Bi, Sk::Uint),
                    0b0111 => (VectorSize::Tri, Sk::Uint),
                    0b1000 => (VectorSize::Quad, Sk::Uint),
                    0b1001 => (VectorSize::Bi, Sk::Bool),
                    0b1010 => (VectorSize::Tri, Sk::Bool),
                    _ => (VectorSize::Quad, Sk::Bool),
                };

                let width = if let Sk::Bool = kind {
                    crate::BOOL_WIDTH
                } else {
                    width
                };

                let ty = || TypeInner::Vector { size, kind, width };
                let args = vec![ty(), ty()];

                let fun = MacroCall::Binary(match name {
                    "equal" => BinaryOperator::Equal,
                    "notEqual" => BinaryOperator::NotEqual,
                    _ => unreachable!(),
                });

                declaration.overloads.push(module.add_builtin(args, fun))
            }
        }
        "min" | "max" => {
            // bits layout
            // bit 0 through 1 - scalar kind
            // bit 2 through 4 - dims
            for bits in 0..0b11100 {
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
            // bit 0 through 1 - dims
            // bit 2 through 4 - types
            //
            // 0b10011 is the last element since splatted single elements
            // were already added
            for bits in 0..0b10011 {
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
            // bit 0 through 1 - float/int/uint
            // bit 2 through 3 - dims
            // bit 4 - splatted
            //
            // 0b11010 is the last element since splatted single elements
            // were already added
            for bits in 0..0b11011 {
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
        "barrier" => declaration
            .overloads
            .push(module.add_builtin(Vec::new(), MacroCall::Barrier)),
        // Add common builtins with floats
        _ => inject_common_builtin(declaration, module, name, 4),
    }
}

/// Injects the builtins into declaration that need doubles
fn inject_double_builtin(declaration: &mut FunctionDeclaration, module: &mut Module, name: &str) {
    let width = 8;
    match name {
        "abs" | "sign" => {
            // bits layout
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 0 through 2 - dims
            for bits in 0..0b111 {
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
            // bit 0 through 1 - dims
            // bit 2 through 3 - splatted/boolean
            //
            // 0b1010 is the last element since splatted with single elements
            // is equal to normal single elements
            for bits in 0..0b1011 {
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
            // bit 0 through 1 - dims
            // bit 2 - splatted
            //
            // 0b110 is the last element since splatted with single elements
            // is equal to normal single elements
            for bits in 0..0b111 {
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
        "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal"
        | "notEqual" => {
            for bits in 0..0b11 {
                let (size, kind) = match bits {
                    0b00 => (VectorSize::Bi, Sk::Float),
                    0b01 => (VectorSize::Tri, Sk::Float),
                    _ => (VectorSize::Quad, Sk::Float),
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
        // Add common builtins with doubles
        _ => inject_common_builtin(declaration, module, name, 8),
    }
}

/// Injects the builtins into declaration that can used either float or doubles
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
                let size = match bits {
                    0b00 => None,
                    0b01 => Some(VectorSize::Bi),
                    0b10 => Some(VectorSize::Tri),
                    _ => Some(VectorSize::Quad),
                };
                let ty = |kind| match size {
                    Some(size) => TypeInner::Vector {
                        size,
                        kind,
                        width: float_width,
                    },
                    None => TypeInner::Scalar {
                        kind,
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

                let second_kind = if fun == MacroCall::MathFunction(MathFunction::Ldexp) {
                    Sk::Sint
                } else {
                    Sk::Float
                };

                declaration
                    .overloads
                    .push(module.add_builtin(vec![ty(Sk::Float), ty(second_kind)], fun))
            }
        }
        "transpose" => {
            // bits layout
            // bit 0 through 3 - dims
            for bits in 0..0b1001 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b11 {
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
            // bit 0 through 2 - dims
            for bits in 0..0b111 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
                    internal: true,
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
            // bit 0 through 3 - dims
            for bits in 0..0b1001 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 0 through 1 - dims
            for bits in 0..0b100 {
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
            // bit 1 through 2 - dims
            for bits in 0..0b1000 {
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
                    MacroCall::SmoothStep { splatted: size },
                ))
            }
        }
        // The function isn't a builtin or we don't yet support it
        _ => {}
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
    TextureSize {
        arrayed: bool,
    },
    ImageLoad {
        multi: bool,
    },
    ImageStore,
    MathFunction(MathFunction),
    FindLsbUint,
    FindMsbUint,
    BitfieldExtract,
    BitfieldInsert,
    Relational(RelationalFunction),
    Unary(UnaryOperator),
    Binary(BinaryOperator),
    Mod(Option<VectorSize>),
    Splatted(MathFunction, Option<VectorSize>, usize),
    MixBoolean,
    Clamp(Option<VectorSize>),
    BitCast(Sk),
    Derivate(Axis, Ctrl),
    Barrier,
    /// SmoothStep needs a separate variant because it might need it's inputs
    /// to be splatted depending on the overload
    SmoothStep {
        /// The size of the splat operation if some
        splatted: Option<VectorSize>,
    },
}

impl MacroCall {
    /// Adds the necessary expressions and statements to the passed body and
    /// finally returns the final expression with the correct result
    pub fn call(
        &self,
        frontend: &mut Frontend,
        ctx: &mut Context,
        body: &mut Block,
        args: &mut [Handle<Expression>],
        meta: Span,
    ) -> Result<Option<Handle<Expression>>> {
        Ok(Some(match *self {
            MacroCall::Sampler => {
                ctx.samplers.insert(args[0], args[1]);
                args[0]
            }
            MacroCall::SamplerShadow => {
                sampled_to_depth(
                    &mut frontend.module,
                    ctx,
                    args[0],
                    meta,
                    &mut frontend.errors,
                );
                frontend.invalidate_expression(ctx, args[0], meta)?;
                ctx.samplers.insert(args[0], args[1]);
                args[0]
            }
            MacroCall::Texture {
                proj,
                offset,
                shadow,
                level_type,
            } => {
                let mut coords = args[1];

                if proj {
                    let size = match *frontend.resolve_type(ctx, coords, meta)? {
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
                    frontend.coordinate_components(ctx, args[0], coords, extra, meta, body)?;

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
                        match frontend.solve_constant(ctx, offset_arg, meta) {
                            Ok(v) => Some(v),
                            Err(e) => {
                                frontend.errors.push(e);
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

                texture_call(ctx, args[0], level, comps, texture_offset, body, meta)?
            }

            MacroCall::TextureSize { arrayed } => {
                let mut expr = ctx.add_expression(
                    Expression::ImageQuery {
                        image: args[0],
                        query: ImageQuery::Size {
                            level: args.get(1).copied(),
                        },
                    },
                    Span::default(),
                    body,
                );

                if arrayed {
                    let mut components = Vec::with_capacity(4);

                    let size = match *frontend.resolve_type(ctx, expr, meta)? {
                        TypeInner::Vector { size: ori_size, .. } => {
                            for index in 0..(ori_size as u32) {
                                components.push(ctx.add_expression(
                                    Expression::AccessIndex { base: expr, index },
                                    Span::default(),
                                    body,
                                ))
                            }

                            match ori_size {
                                VectorSize::Bi => VectorSize::Tri,
                                _ => VectorSize::Quad,
                            }
                        }
                        _ => {
                            components.push(expr);
                            VectorSize::Bi
                        }
                    };

                    components.push(ctx.add_expression(
                        Expression::ImageQuery {
                            image: args[0],
                            query: ImageQuery::NumLayers,
                        },
                        Span::default(),
                        body,
                    ));

                    let ty = frontend.module.types.insert(
                        Type {
                            name: None,
                            inner: TypeInner::Vector {
                                size,
                                kind: crate::ScalarKind::Uint,
                                width: 4,
                            },
                        },
                        Span::default(),
                    );

                    expr = ctx.add_expression(Expression::Compose { components, ty }, meta, body)
                }

                ctx.add_expression(
                    Expression::As {
                        expr,
                        kind: Sk::Sint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::ImageLoad { multi } => {
                let comps =
                    frontend.coordinate_components(ctx, args[0], args[1], None, meta, body)?;
                let (sample, level) = match (multi, args.get(2)) {
                    (_, None) => (None, None),
                    (true, Some(&arg)) => (Some(arg), None),
                    (false, Some(&arg)) => (None, Some(arg)),
                };
                ctx.add_expression(
                    Expression::ImageLoad {
                        image: args[0],
                        coordinate: comps.coordinate,
                        array_index: comps.array_index,
                        sample,
                        level,
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::ImageStore => {
                let comps =
                    frontend.coordinate_components(ctx, args[0], args[1], None, meta, body)?;
                ctx.emit_restart(body);
                body.push(
                    crate::Statement::ImageStore {
                        image: args[0],
                        coordinate: comps.coordinate,
                        array_index: comps.array_index,
                        value: args[2],
                    },
                    meta,
                );
                return Ok(None);
            }
            MacroCall::MathFunction(fun) => ctx.add_expression(
                Expression::Math {
                    fun,
                    arg: args[0],
                    arg1: args.get(1).copied(),
                    arg2: args.get(2).copied(),
                    arg3: args.get(3).copied(),
                },
                Span::default(),
                body,
            ),
            mc @ (MacroCall::FindLsbUint | MacroCall::FindMsbUint) => {
                let fun = match mc {
                    MacroCall::FindLsbUint => MathFunction::FindLsb,
                    MacroCall::FindMsbUint => MathFunction::FindMsb,
                    _ => unreachable!(),
                };
                let res = ctx.add_expression(
                    Expression::Math {
                        fun,
                        arg: args[0],
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    },
                    Span::default(),
                    body,
                );
                ctx.add_expression(
                    Expression::As {
                        expr: res,
                        kind: Sk::Sint,
                        convert: Some(4),
                    },
                    Span::default(),
                    body,
                )
            }
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
                ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::InsertBits,
                        arg: args[0],
                        arg1: Some(args[1]),
                        arg2: Some(conv_arg_2),
                        arg3: Some(conv_arg_3),
                    },
                    Span::default(),
                    body,
                )
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
                ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::ExtractBits,
                        arg: args[0],
                        arg1: Some(conv_arg_1),
                        arg2: Some(conv_arg_2),
                        arg3: None,
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::Relational(fun) => ctx.add_expression(
                Expression::Relational {
                    fun,
                    argument: args[0],
                },
                Span::default(),
                body,
            ),
            MacroCall::Unary(op) => ctx.add_expression(
                Expression::Unary { op, expr: args[0] },
                Span::default(),
                body,
            ),
            MacroCall::Binary(op) => ctx.add_expression(
                Expression::Binary {
                    op,
                    left: args[0],
                    right: args[1],
                },
                Span::default(),
                body,
            ),
            MacroCall::Mod(size) => {
                ctx.implicit_splat(frontend, &mut args[1], meta, size)?;

                // x - y * floor(x / y)

                let div = ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: args[0],
                        right: args[1],
                    },
                    Span::default(),
                    body,
                );
                let floor = ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::Floor,
                        arg: div,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    },
                    Span::default(),
                    body,
                );
                let mult = ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: floor,
                        right: args[1],
                    },
                    Span::default(),
                    body,
                );
                ctx.add_expression(
                    Expression::Binary {
                        op: BinaryOperator::Subtract,
                        left: args[0],
                        right: mult,
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::Splatted(fun, size, i) => {
                ctx.implicit_splat(frontend, &mut args[i], meta, size)?;

                ctx.add_expression(
                    Expression::Math {
                        fun,
                        arg: args[0],
                        arg1: args.get(1).copied(),
                        arg2: args.get(2).copied(),
                        arg3: args.get(3).copied(),
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::MixBoolean => ctx.add_expression(
                Expression::Select {
                    condition: args[2],
                    accept: args[1],
                    reject: args[0],
                },
                Span::default(),
                body,
            ),
            MacroCall::Clamp(size) => {
                ctx.implicit_splat(frontend, &mut args[1], meta, size)?;
                ctx.implicit_splat(frontend, &mut args[2], meta, size)?;

                ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: args[0],
                        arg1: args.get(1).copied(),
                        arg2: args.get(2).copied(),
                        arg3: args.get(3).copied(),
                    },
                    Span::default(),
                    body,
                )
            }
            MacroCall::BitCast(kind) => ctx.add_expression(
                Expression::As {
                    expr: args[0],
                    kind,
                    convert: None,
                },
                Span::default(),
                body,
            ),
            MacroCall::Derivate(axis, ctrl) => ctx.add_expression(
                Expression::Derivative {
                    axis,
                    ctrl,
                    expr: args[0],
                },
                Span::default(),
                body,
            ),
            MacroCall::Barrier => {
                ctx.emit_restart(body);
                body.push(crate::Statement::Barrier(crate::Barrier::all()), meta);
                return Ok(None);
            }
            MacroCall::SmoothStep { splatted } => {
                ctx.implicit_splat(frontend, &mut args[0], meta, splatted)?;
                ctx.implicit_splat(frontend, &mut args[1], meta, splatted)?;

                ctx.add_expression(
                    Expression::Math {
                        fun: MathFunction::SmoothStep,
                        arg: args[0],
                        arg1: args.get(1).copied(),
                        arg2: args.get(2).copied(),
                        arg3: None,
                    },
                    Span::default(),
                    body,
                )
            }
        }))
    }
}

fn texture_call(
    ctx: &mut Context,
    image: Handle<Expression>,
    level: SampleLevel,
    comps: CoordComponents,
    offset: Option<Handle<Expression>>,
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
                gather: None, //TODO
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
/// Obtained by calling [`coordinate_components`](Frontend::coordinate_components)
#[derive(Debug)]
struct CoordComponents {
    coordinate: Handle<Expression>,
    depth_ref: Option<Handle<Expression>>,
    array_index: Option<Handle<Expression>>,
    used_extra: bool,
}

impl Frontend {
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
                Dim::D1 => None,
                Dim::D2 => Some(VectorSize::Bi),
                Dim::D3 => Some(VectorSize::Tri),
                Dim::Cube => Some(VectorSize::Tri),
            };
            let coord_size = match *self.resolve_type(ctx, coord, meta)? {
                TypeInner::Vector { size, .. } => Some(size),
                _ => None,
            };
            let (shadow, storage) = match class {
                ImageClass::Depth { .. } => (true, false),
                ImageClass::Storage { .. } => (false, true),
                ImageClass::Sampled { .. } => (false, false),
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

            let array_index = if arrayed && !(storage && dim == Dim::Cube) {
                let index = coord_index;
                coord_index += 1;

                Some(ctx.add_expression(
                    Expression::AccessIndex { base: coord, index },
                    Span::default(),
                    body,
                ))
            } else {
                None
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
    // Get the a mutable type handle of the underlying image storage
    let ty = match ctx[image] {
        Expression::GlobalVariable(handle) => &mut module.global_variables.get_mut(handle).ty,
        Expression::FunctionArgument(i) => {
            // Mark the function argument as carrying a depth texture
            ctx.parameters_info[i as usize].depth = true;
            // NOTE: We need to later also change the parameter type
            &mut ctx.arguments[i as usize].ty
        }
        _ => {
            // Only globals and function arguments are allowed to carry an image
            return errors.push(Error {
                kind: ErrorKind::SemanticError("Not a valid texture expression".into()),
                meta,
            });
        }
    };

    match module.types[*ty].inner {
        // Update the image class to depth in case it already isn't
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
            // Other image classes aren't allowed to be transformed to depth
            ImageClass::Storage { .. } => errors.push(Error {
                kind: ErrorKind::SemanticError("Not a texture".into()),
                meta,
            }),
        },
        _ => errors.push(Error {
            kind: ErrorKind::SemanticError("Not a texture".into()),
            meta,
        }),
    };

    // Copy the handle to allow borrowing the `ctx` again
    let ty = *ty;

    // If the image was passed through a function argument we also need to change
    // the corresponding parameter
    if let Expression::FunctionArgument(i) = ctx[image] {
        ctx.parameters[i as usize] = ty;
    }
}

bitflags::bitflags! {
    /// Influences the operation `texture_args_generator`
    struct TextureArgsOptions: u32 {
        /// Generates multisampled variants of images
        const MULTI = 1 << 0;
        /// Generates shadow variants of images
        const SHADOW = 1 << 1;
        /// Generates standard images
        const STANDARD = 1 << 2;
        /// Generates cube arrayed images
        const CUBE_ARRAY = 1 << 3;
        /// Generates cube arrayed images
        const D2_MULTI_ARRAY = 1 << 4;
    }
}

impl From<BuiltinVariations> for TextureArgsOptions {
    fn from(variations: BuiltinVariations) -> Self {
        let mut options = TextureArgsOptions::empty();
        if variations.contains(BuiltinVariations::STANDARD) {
            options |= TextureArgsOptions::STANDARD
        }
        if variations.contains(BuiltinVariations::CUBE_TEXTURES_ARRAY) {
            options |= TextureArgsOptions::CUBE_ARRAY
        }
        if variations.contains(BuiltinVariations::D2_MULTI_TEXTURES_ARRAY) {
            options |= TextureArgsOptions::D2_MULTI_ARRAY
        }
        options
    }
}

/// Helper function to generate the image components for texture/image builtins
///
/// Calls the passed function `f` with:
/// ```text
/// f(ScalarKind, ImageDimension, arrayed, multi, shadow)
/// ```
///
/// `options` controls extra image variants generation like multisampling and depth,
/// see the struct documentation
fn texture_args_generator(
    options: TextureArgsOptions,
    mut f: impl FnMut(crate::ScalarKind, Dim, bool, bool, bool),
) {
    for kind in [Sk::Float, Sk::Uint, Sk::Sint].iter().copied() {
        for dim in [Dim::D1, Dim::D2, Dim::D3, Dim::Cube].iter().copied() {
            for arrayed in [false, true].iter().copied() {
                if dim == Dim::Cube && arrayed {
                    if !options.contains(TextureArgsOptions::CUBE_ARRAY) {
                        continue;
                    }
                } else if Dim::D2 == dim
                    && options.contains(TextureArgsOptions::MULTI)
                    && arrayed
                    && options.contains(TextureArgsOptions::D2_MULTI_ARRAY)
                {
                    // multisampling for sampler2DMSArray
                    f(kind, dim, arrayed, true, false);
                } else if !options.contains(TextureArgsOptions::STANDARD) {
                    continue;
                }

                f(kind, dim, arrayed, false, false);

                // 3D images can't be neither arrayed nor shadow
                // so we break out early, this way arrayed will always
                // be false and we won't hit the shadow branch
                if let Dim::D3 = dim {
                    break;
                }

                if Dim::D2 == dim && options.contains(TextureArgsOptions::MULTI) && !arrayed {
                    // multisampling
                    f(kind, dim, arrayed, true, false);
                }

                if Sk::Float == kind && options.contains(TextureArgsOptions::SHADOW) {
                    // shadow
                    f(kind, dim, arrayed, false, true);
                }
            }
        }
    }
}

/// Helper functions used to convert from a image dimension into a integer representing the
/// number of components needed for the coordinates vector (1 means scalar instead of vector)
const fn image_dims_to_coords_size(dim: Dim) -> usize {
    match dim {
        Dim::D1 => 1,
        Dim::D2 => 2,
        _ => 3,
    }
}
