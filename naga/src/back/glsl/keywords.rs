pub const RESERVED_KEYWORDS: &[&str] = &[
    //
    // GLSL 4.6 keywords, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L2004-L2322
    // GLSL ES 3.2 keywords, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/es/3.2/GLSL_ES_Specification_3.20.html#L2166-L2478
    //
    // Note: The GLSL ES 3.2 keywords are the same as GLSL 4.6 keywords with some residing in the reserved section.
    // The only exception are the missing Vulkan keywords which I think is an oversight (see https://github.com/KhronosGroup/OpenGL-Registry/issues/585).
    //
    "const",
    "uniform",
    "buffer",
    "shared",
    "attribute",
    "varying",
    "coherent",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
    "atomic_uint",
    "layout",
    "centroid",
    "flat",
    "smooth",
    "noperspective",
    "patch",
    "sample",
    "invariant",
    "precise",
    "break",
    "continue",
    "do",
    "for",
    "while",
    "switch",
    "case",
    "default",
    "if",
    "else",
    "subroutine",
    "in",
    "out",
    "inout",
    "int",
    "void",
    "bool",
    "true",
    "false",
    "float",
    "double",
    "discard",
    "return",
    "vec2",
    "vec3",
    "vec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "bvec2",
    "bvec3",
    "bvec4",
    "uint",
    "uvec2",
    "uvec3",
    "uvec4",
    "dvec2",
    "dvec3",
    "dvec4",
    "mat2",
    "mat3",
    "mat4",
    "mat2x2",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x3",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "mat4x4",
    "dmat2",
    "dmat3",
    "dmat4",
    "dmat2x2",
    "dmat2x3",
    "dmat2x4",
    "dmat3x2",
    "dmat3x3",
    "dmat3x4",
    "dmat4x2",
    "dmat4x3",
    "dmat4x4",
    "lowp",
    "mediump",
    "highp",
    "precision",
    "sampler1D",
    "sampler1DShadow",
    "sampler1DArray",
    "sampler1DArrayShadow",
    "isampler1D",
    "isampler1DArray",
    "usampler1D",
    "usampler1DArray",
    "sampler2D",
    "sampler2DShadow",
    "sampler2DArray",
    "sampler2DArrayShadow",
    "isampler2D",
    "isampler2DArray",
    "usampler2D",
    "usampler2DArray",
    "sampler2DRect",
    "sampler2DRectShadow",
    "isampler2DRect",
    "usampler2DRect",
    "sampler2DMS",
    "isampler2DMS",
    "usampler2DMS",
    "sampler2DMSArray",
    "isampler2DMSArray",
    "usampler2DMSArray",
    "sampler3D",
    "isampler3D",
    "usampler3D",
    "samplerCube",
    "samplerCubeShadow",
    "isamplerCube",
    "usamplerCube",
    "samplerCubeArray",
    "samplerCubeArrayShadow",
    "isamplerCubeArray",
    "usamplerCubeArray",
    "samplerBuffer",
    "isamplerBuffer",
    "usamplerBuffer",
    "image1D",
    "iimage1D",
    "uimage1D",
    "image1DArray",
    "iimage1DArray",
    "uimage1DArray",
    "image2D",
    "iimage2D",
    "uimage2D",
    "image2DArray",
    "iimage2DArray",
    "uimage2DArray",
    "image2DRect",
    "iimage2DRect",
    "uimage2DRect",
    "image2DMS",
    "iimage2DMS",
    "uimage2DMS",
    "image2DMSArray",
    "iimage2DMSArray",
    "uimage2DMSArray",
    "image3D",
    "iimage3D",
    "uimage3D",
    "imageCube",
    "iimageCube",
    "uimageCube",
    "imageCubeArray",
    "iimageCubeArray",
    "uimageCubeArray",
    "imageBuffer",
    "iimageBuffer",
    "uimageBuffer",
    "struct",
    // Vulkan keywords
    "texture1D",
    "texture1DArray",
    "itexture1D",
    "itexture1DArray",
    "utexture1D",
    "utexture1DArray",
    "texture2D",
    "texture2DArray",
    "itexture2D",
    "itexture2DArray",
    "utexture2D",
    "utexture2DArray",
    "texture2DRect",
    "itexture2DRect",
    "utexture2DRect",
    "texture2DMS",
    "itexture2DMS",
    "utexture2DMS",
    "texture2DMSArray",
    "itexture2DMSArray",
    "utexture2DMSArray",
    "texture3D",
    "itexture3D",
    "utexture3D",
    "textureCube",
    "itextureCube",
    "utextureCube",
    "textureCubeArray",
    "itextureCubeArray",
    "utextureCubeArray",
    "textureBuffer",
    "itextureBuffer",
    "utextureBuffer",
    "sampler",
    "samplerShadow",
    "subpassInput",
    "isubpassInput",
    "usubpassInput",
    "subpassInputMS",
    "isubpassInputMS",
    "usubpassInputMS",
    // Reserved keywords
    "common",
    "partition",
    "active",
    "asm",
    "class",
    "union",
    "enum",
    "typedef",
    "template",
    "this",
    "resource",
    "goto",
    "inline",
    "noinline",
    "public",
    "static",
    "extern",
    "external",
    "interface",
    "long",
    "short",
    "half",
    "fixed",
    "unsigned",
    "superp",
    "input",
    "output",
    "hvec2",
    "hvec3",
    "hvec4",
    "fvec2",
    "fvec3",
    "fvec4",
    "filter",
    "sizeof",
    "cast",
    "namespace",
    "using",
    "sampler3DRect",
    //
    // GLSL 4.6 Built-In Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L13314
    //
    // Angle and Trigonometry Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L13469-L13561C5
    "radians",
    "degrees",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    // Exponential Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L13569-L13620
    "pow",
    "exp",
    "log",
    "exp2",
    "log2",
    "sqrt",
    "inversesqrt",
    // Common Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L13628-L13908
    "abs",
    "sign",
    "floor",
    "trunc",
    "round",
    "roundEven",
    "ceil",
    "fract",
    "mod",
    "modf",
    "min",
    "max",
    "clamp",
    "mix",
    "step",
    "smoothstep",
    "isnan",
    "isinf",
    "floatBitsToInt",
    "floatBitsToUint",
    "intBitsToFloat",
    "uintBitsToFloat",
    "fma",
    "frexp",
    "ldexp",
    // Floating-Point Pack and Unpack Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L13916-L14007
    "packUnorm2x16",
    "packSnorm2x16",
    "packUnorm4x8",
    "packSnorm4x8",
    "unpackUnorm2x16",
    "unpackSnorm2x16",
    "unpackUnorm4x8",
    "unpackSnorm4x8",
    "packHalf2x16",
    "unpackHalf2x16",
    "packDouble2x32",
    "unpackDouble2x32",
    // Geometric Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14014-L14121
    "length",
    "distance",
    "dot",
    "cross",
    "normalize",
    "ftransform",
    "faceforward",
    "reflect",
    "refract",
    // Matrix Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14151-L14215
    "matrixCompMult",
    "outerProduct",
    "transpose",
    "determinant",
    "inverse",
    // Vector Relational Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14259-L14322
    "lessThan",
    "lessThanEqual",
    "greaterThan",
    "greaterThanEqual",
    "equal",
    "notEqual",
    "any",
    "all",
    "not",
    // Integer Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14335-L14432
    "uaddCarry",
    "usubBorrow",
    "umulExtended",
    "imulExtended",
    "bitfieldExtract",
    "bitfieldInsert",
    "bitfieldReverse",
    "bitCount",
    "findLSB",
    "findMSB",
    // Texture Query Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14645-L14732
    "textureSize",
    "textureQueryLod",
    "textureQueryLevels",
    "textureSamples",
    // Texel Lookup Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L14736-L14997
    "texture",
    "textureProj",
    "textureLod",
    "textureOffset",
    "texelFetch",
    "texelFetchOffset",
    "textureProjOffset",
    "textureLodOffset",
    "textureProjLod",
    "textureProjLodOffset",
    "textureGrad",
    "textureGradOffset",
    "textureProjGrad",
    "textureProjGradOffset",
    // Texture Gather Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15077-L15154
    "textureGather",
    "textureGatherOffset",
    "textureGatherOffsets",
    // Compatibility Profile Texture Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15161-L15220
    "texture1D",
    "texture1DProj",
    "texture1DLod",
    "texture1DProjLod",
    "texture2D",
    "texture2DProj",
    "texture2DLod",
    "texture2DProjLod",
    "texture3D",
    "texture3DProj",
    "texture3DLod",
    "texture3DProjLod",
    "textureCube",
    "textureCubeLod",
    "shadow1D",
    "shadow2D",
    "shadow1DProj",
    "shadow2DProj",
    "shadow1DLod",
    "shadow2DLod",
    "shadow1DProjLod",
    "shadow2DProjLod",
    // Atomic Counter Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15241-L15531
    "atomicCounterIncrement",
    "atomicCounterDecrement",
    "atomicCounter",
    "atomicCounterAdd",
    "atomicCounterSubtract",
    "atomicCounterMin",
    "atomicCounterMax",
    "atomicCounterAnd",
    "atomicCounterOr",
    "atomicCounterXor",
    "atomicCounterExchange",
    "atomicCounterCompSwap",
    // Atomic Memory Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15563-L15624
    "atomicAdd",
    "atomicMin",
    "atomicMax",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompSwap",
    // Image Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15763-L15878
    "imageSize",
    "imageSamples",
    "imageLoad",
    "imageStore",
    "imageAtomicAdd",
    "imageAtomicMin",
    "imageAtomicMax",
    "imageAtomicAnd",
    "imageAtomicOr",
    "imageAtomicXor",
    "imageAtomicExchange",
    "imageAtomicCompSwap",
    // Geometry Shader Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L15886-L15932
    "EmitStreamVertex",
    "EndStreamPrimitive",
    "EmitVertex",
    "EndPrimitive",
    // Fragment Processing Functions, Derivative Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16041-L16114
    "dFdx",
    "dFdy",
    "dFdxFine",
    "dFdyFine",
    "dFdxCoarse",
    "dFdyCoarse",
    "fwidth",
    "fwidthFine",
    "fwidthCoarse",
    // Fragment Processing Functions, Interpolation Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16150-L16198
    "interpolateAtCentroid",
    "interpolateAtSample",
    "interpolateAtOffset",
    // Noise Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16214-L16243
    "noise1",
    "noise2",
    "noise3",
    "noise4",
    // Shader Invocation Control Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16255-L16276
    "barrier",
    // Shader Memory Control Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16336-L16382
    "memoryBarrier",
    "memoryBarrierAtomicCounter",
    "memoryBarrierBuffer",
    "memoryBarrierShared",
    "memoryBarrierImage",
    "groupMemoryBarrier",
    // Subpass-Input Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16451-L16470
    "subpassLoad",
    // Shader Invocation Group Functions, from https://github.com/KhronosGroup/OpenGL-Registry/blob/d00e11dc1a1ffba581d633f21f70202051248d5c/specs/gl/GLSLangSpec.4.60.html#L16483-L16511
    "anyInvocation",
    "allInvocations",
    "allInvocationsEqual",
    //
    // entry point name (should not be shadowed)
    //
    "main",
    // Naga utilities:
    super::MODF_FUNCTION,
    super::FREXP_FUNCTION,
    super::BASE_INSTANCE_BINDING,
];
