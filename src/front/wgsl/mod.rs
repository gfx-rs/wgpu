//! Front end for consuming [WebGPU Shading Language][wgsl].
//!
//! [wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html

mod conv;
mod lexer;
#[cfg(test)]
mod tests;

use crate::{
    arena::{Arena, Handle},
    proc::{
        ensure_block_returns, Alignment, Layouter, ResolveContext, ResolveError, TypeResolution,
    },
    ConstantInner, FastHashMap, ScalarValue,
};

use self::lexer::Lexer;
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::{Files, SimpleFile},
    term::{
        self,
        termcolor::{ColorChoice, ColorSpec, StandardStream, WriteColor},
    },
};
use std::{
    borrow::Cow,
    convert::TryFrom,
    io::{self, Write},
    iter,
    num::{NonZeroU32, ParseFloatError, ParseIntError},
    ops,
};
use thiserror::Error;

type Span = ops::Range<usize>;
type TokenSpan<'a> = (Token<'a>, Span);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    DoubleColon,
    Paren(char),
    DoubleParen(char),
    Number {
        value: &'a str,
        ty: char,
        width: &'a str,
    },
    String(&'a str),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    Arrow,
    Unknown(char),
    UnterminatedString,
    Trivia,
    End,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    Float,
    Uint,
    Sint,
    Constant,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: ']]', ','
    AttributeSeparator,
    /// Expected: '}', identifier
    FieldName,
    /// Expected: ']]', 'access', 'stride'
    TypeAttribute,
    /// Expected: ';', '{', word
    Statement,
    /// Expected: 'case', 'default', '}'
    SwitchItem,
    /// Expected: ',', ')'
    WorkgroupSizeSeparator,
    /// Expected: 'struct', 'let', 'var', 'type', ';', 'fn', eof
    GlobalItem,
    /// Expected: ']]', 'size', 'align'
    StructAttribute,
}

#[derive(Clone, Debug, Error)]
pub enum Error<'a> {
    #[error("")]
    Unexpected(TokenSpan<'a>, ExpectedToken<'a>),
    #[error("")]
    BadU32(Span, ParseIntError),
    #[error("")]
    BadI32(Span, ParseIntError),
    #[error("")]
    BadFloat(Span, ParseFloatError),
    #[error("")]
    BadU32Constant(Span),
    #[error("")]
    BadScalarWidth(Span, &'a str),
    #[error("")]
    BadAccessor(Span),
    #[error("bad texture`")]
    BadTexture(Span),
    #[error("bad texture coordinate")]
    BadCoordinate,
    #[error("invalid type cast")]
    BadTypeCast {
        span: Span,
        from_type: String,
        to_type: String,
    },
    #[error("bad texture sample type. Only f32, i32 and u32 are valid")]
    BadTextureSampleType {
        span: Span,
        kind: crate::ScalarKind,
        width: u8,
    },
    #[error(transparent)]
    InvalidResolve(ResolveError),
    #[error("for(;;) initializer is not an assignment or a function call")]
    InvalidForInitializer(Span),
    #[error("resource type {0:?} is invalid")]
    InvalidResourceType(Handle<crate::Type>),
    #[error("unknown import: `{0}`")]
    UnknownImport(&'a str),
    #[error("unknown storage class")]
    UnknownStorageClass(Span),
    #[error("unknown attribute")]
    UnknownAttribute(Span),
    #[error("unknown scalar kind: `{0}`")]
    UnknownScalarKind(&'a str),
    #[error("unknown builtin")]
    UnknownBuiltin(Span),
    #[error("unknown access: `{0}`")]
    UnknownAccess(&'a str),
    #[error("unknown shader stage")]
    UnknownShaderStage(Span),
    #[error("unknown identifier: `{1}`")]
    UnknownIdent(Span, &'a str),
    #[error("unknown scalar type")]
    UnknownScalarType(Span),
    #[error("unknown type")]
    UnknownType(Span),
    #[error("unknown function: `{0}`")]
    UnknownFunction(&'a str),
    #[error("unknown storage format")]
    UnknownStorageFormat(Span),
    #[error("unknown conservative depth")]
    UnknownConservativeDepth(Span),
    #[error("array stride must not be 0")]
    ZeroStride(Span),
    #[error("struct member size or alignment must not be 0")]
    ZeroSizeOrAlign(Span),
    #[error("not a composite type: {0:?}")]
    NotCompositeType(Handle<crate::Type>),
    #[error("Input/output binding is not consistent: location {0:?}, built-in {1:?}, interpolation {2:?}, and sampling {3:?}")]
    InconsistentBinding(
        Option<u32>,
        Option<crate::BuiltIn>,
        Option<crate::Interpolation>,
        Option<crate::Sampling>,
    ),
    #[error("call to local `{0}(..)` can't be resolved")]
    UnknownLocalFunction(&'a str),
    #[error("builtin {0:?} is not implemented")]
    UnimplementedBuiltin(crate::BuiltIn),
    #[error("expression {0} doesn't match its given type {1:?}")]
    LetTypeMismatch(&'a str, Handle<crate::Type>),
    #[error("other error")]
    Other,
}

impl<'a> Error<'a> {
    fn as_parse_error(&self, source: &'a str) -> ParseError {
        match *self {
            Error::Unexpected((_, ref unexpected_span), expected) => {
                let expected_str = match expected {
                        ExpectedToken::Token(token) => {
                            match token {
                                Token::Separator(c) => format!("'{}'", c),
                                Token::DoubleColon => "'::'".to_string(),
                                Token::Paren(c) => format!("'{}'", c),
                                Token::DoubleParen(c) => format!("'{}{}'", c, c),
                                Token::Number { value, .. } => {
                                    format!("number ({})", value)
                                }
                                Token::String(s) => format!("string literal ('{}')", s.to_string()),
                                Token::Word(s) => s.to_string(),
                                Token::Operation(c) => format!("operation ('{}')", c),
                                Token::LogicalOperation(c) => format!("logical operation ('{}')", c),
                                Token::ShiftOperation(c) => format!("bitshift ('{}{}')", c, c),
                                Token::Arrow => "->".to_string(),
                                Token::Unknown(c) => format!("unkown ('{}')", c),
                                Token::UnterminatedString => "unterminated string".to_string(),
                                Token::Trivia => "trivia".to_string(),
                                Token::End => "end".to_string(),
                            }
                        }
                        ExpectedToken::Identifier => "identifier".to_string(),
                        ExpectedToken::Float => "floating point literal".to_string(),
                        ExpectedToken::Uint => "non-negative integer literal".to_string(),
                        ExpectedToken::Sint => "integer literal".to_string(),
                        ExpectedToken::Constant => "constant".to_string(),
                        ExpectedToken::PrimaryExpression => "expression".to_string(),
                        ExpectedToken::AttributeSeparator => "attribute separator (',') or an end of the attribute list (']]')".to_string(),
                        ExpectedToken::FieldName => "field name or a closing curly bracket to signify the end of the struct".to_string(),
                        ExpectedToken::TypeAttribute => "type attribute ('access' or 'stride') or and of the attribute list (']]')".to_string(),
                        ExpectedToken::Statement => "statement".to_string(),
                        ExpectedToken::SwitchItem => "switch item ('case' or 'default') or a closing curly bracket to signify the end of the switch statement ('}')".to_string(),
                        ExpectedToken::WorkgroupSizeSeparator => "workgroup size separator (',') or a closing parenthesis".to_string(),
                        ExpectedToken::GlobalItem => "global item ('struct', 'let', 'var', 'type', ';', 'fn') or the end of the file".to_string(),
                        ExpectedToken::StructAttribute => "struct attribute ('size' or 'align') or an end of the attribute list (']]')".to_string(),
                    };
                    ParseError {
                    message: format!(
                        "expected {}, found '{}'",
                        expected_str,
                        &source[unexpected_span.clone()],
                    ),
                    labels: vec![(
                        unexpected_span.clone(),
                        format!("expected {}", expected_str).into(),
                    )],
                    notes: vec![],
                }
            },
            Error::BadU32(ref bad_span, ref err) => ParseError {
                message: format!(
                    "expected non-negative integer literal, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected positive integer".into())],
                notes: vec![err.to_string()],
            },
            Error::BadI32(ref bad_span, ref err) => ParseError {
                message: format!(
                    "expected integer literal, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected integer".into())],
                notes: vec![err.to_string()],
            },
            Error::BadFloat(ref bad_span, ref err) => ParseError {
                message: format!(
                    "expected floating-point literal, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected floating-point literal".into())],
                notes: vec![err.to_string()],
            },
            Error::BadU32Constant(ref bad_span) => ParseError {
                message: format!(
                    "expected non-negative integer constant expression, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected non-negative integer".into())],
                notes: vec![],
            },

            Error::BadScalarWidth(ref bad_span, width) => ParseError {
                message: format!("invalid width of `{}` for literal", width,),
                labels: vec![(bad_span.clone(), "invalid width".into())],
                notes: vec!["valid widths are 8, 16, 32, 64".to_string()],
            },
            Error::BadAccessor(ref accessor_span) => ParseError {
                message: format!(
                    "invalid field accessor `{}`",
                    &source[accessor_span.clone()],
                ),
                labels: vec![(accessor_span.clone(), "invalid accessor".into())],
                notes: vec![],
            },
            Error::UnknownIdent(ref ident_span, ident) => ParseError {
                message: format!("no definition in scope for identifier: '{}'", ident),
                labels: vec![(ident_span.clone(), "unknown identifier".into())],
                notes: vec![],
            },
            Error::UnknownScalarType(ref bad_span) => ParseError {
                message: format!("unknown scalar type: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown scalar type".into())],
                notes: vec!["Valid scalar types are f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool".into()],
            },
            Error::BadTextureSampleType { ref span, kind, width } => ParseError {
                message: format!("texture sample type must be one of f32, i32 or u32, but found {}", kind.to_wgsl(width)),
                labels: vec![(span.clone(), "must be one of f32, i32 or u32".into())],
                notes: vec![],
            },
            Error::BadTexture(ref bad_span) => ParseError {
                message: format!("expected an image, but found '{}' which is not an image", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "not an image".into())],
                notes: vec![],
            },
            Error::BadTypeCast { ref span, ref from_type, ref to_type } => {
                let msg = format!("cannot cast a {} to a {}", from_type, to_type);
                ParseError {
                    message: msg.clone(),
                    labels: vec![(span.clone(), msg.into())],
                    notes: vec![],
                }
            },
            Error::InvalidForInitializer(ref bad_span) => ParseError {
                message: format!("for(;;) initializer is not an assignment or a function call: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "not an assignment or function call".into())],
                notes: vec![],
            },
            Error::UnknownStorageClass(ref bad_span) => ParseError {
                message: format!("unknown storage class: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown storage class".into())],
                notes: vec![],
            },
            Error::UnknownAttribute(ref bad_span) => ParseError {
                message: format!("unknown attribute: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown attribute".into())],
                notes: vec![],
            },
            Error::UnknownBuiltin(ref bad_span) => ParseError {
                message: format!("unknown builtin: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown builtin".into())],
                notes: vec![],
            },
            Error::UnknownShaderStage(ref bad_span) => ParseError {
                message: format!("unknown shader stage: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown shader stage".into())],
                notes: vec![],
            },
            Error::UnknownStorageFormat(ref bad_span) => ParseError {
                message: format!("unknown storage format: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown storage format".into())],
                notes: vec![],
            },
            Error::UnknownConservativeDepth(ref bad_span) => ParseError {
                message: format!("unknown conservative depth: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown conservative depth".into())],
                notes: vec![],
            },
            Error::UnknownType(ref bad_span) => ParseError {
                message: format!("unknown type: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown type".into())],
                notes: vec![],
            },
            Error::ZeroStride(ref bad_span) => ParseError {
                message: "array stride must not be zero".to_string(),
                labels: vec![(bad_span.clone(), "array stride must not be zero".into())],
                notes: vec![],
            },
            Error::ZeroSizeOrAlign(ref bad_span) => ParseError {
                message: "struct member size or alignment must not be 0".to_string(),
                labels: vec![(bad_span.clone(), "struct member size or alignment must not be 0".into())],
                notes: vec![],
            },

            ref error => ParseError {
                message: error.to_string(),
                labels: vec![],
                notes: vec![],
            },
        }
    }
}

impl crate::StorageFormat {
    pub fn to_wgsl(self) -> &'static str {
        use crate::StorageFormat as Sf;
        match self {
            Sf::R8Unorm => "r8unorm",
            Sf::R8Snorm => "r8snorm",
            Sf::R8Uint => "r8uint",
            Sf::R8Sint => "r8sint",
            Sf::R16Uint => "r16uint",
            Sf::R16Sint => "r16sint",
            Sf::R16Float => "r16float",
            Sf::Rg8Unorm => "rg8unorm",
            Sf::Rg8Snorm => "rg8snorm",
            Sf::Rg8Uint => "rg8uint",
            Sf::Rg8Sint => "rg8sint",
            Sf::R32Uint => "r32uint",
            Sf::R32Sint => "r32sint",
            Sf::R32Float => "r32float",
            Sf::Rg16Uint => "rg16uint",
            Sf::Rg16Sint => "rg16sint",
            Sf::Rg16Float => "rg16float",
            Sf::Rgba8Unorm => "rgba8unorm",
            Sf::Rgba8Snorm => "rgba8snorm",
            Sf::Rgba8Uint => "rgba8uint",
            Sf::Rgba8Sint => "rgba8sint",
            Sf::Rgb10a2Unorm => "rgb10a2unorm",
            Sf::Rg11b10Float => "rg11b10float",
            Sf::Rg32Uint => "rg32uint",
            Sf::Rg32Sint => "rg32sint",
            Sf::Rg32Float => "rg32float",
            Sf::Rgba16Uint => "rgba16uint",
            Sf::Rgba16Sint => "rgba16sint",
            Sf::Rgba16Float => "rgba16float",
            Sf::Rgba32Uint => "rgba32uint",
            Sf::Rgba32Sint => "rgba32sint",
            Sf::Rgba32Float => "rgba32float",
        }
    }
}

impl crate::TypeInner {
    /// Formats the type as it is written in wgsl.
    ///
    /// For example `vec3<f32>`.
    ///
    /// Note: The names of a `TypeInner::Struct` is not known. Therefore this method will simply return "struct" for them.
    pub fn to_wgsl(
        &self,
        types: &Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> String {
        match *self {
            crate::TypeInner::Scalar { kind, width } => kind.to_wgsl(width),
            crate::TypeInner::Vector { size, kind, width } => {
                format!("vec{}<{}>", size as u32, kind.to_wgsl(width))
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                format!(
                    "mat{}x{}<{}>",
                    columns as u32,
                    rows as u32,
                    crate::ScalarKind::Float.to_wgsl(width),
                )
            }
            crate::TypeInner::Pointer { base, .. } => {
                let base = &types[base];
                let name = base.name.as_deref().unwrap_or("unknown");
                format!("*{}", name)
            }
            crate::TypeInner::ValuePointer { kind, width, .. } => {
                format!("*{}", kind.to_wgsl(width))
            }
            crate::TypeInner::Array { base, size, .. } => {
                let member_type = &types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => {
                        let size = constants[size].name.as_deref().unwrap_or("unknown");
                        format!("{}[{}]", base, size)
                    }
                    crate::ArraySize::Dynamic => format!("{}[]", base),
                }
            }
            crate::TypeInner::Struct { .. } => {
                // TODO: Actually output the struct?
                "struct".to_string()
            }
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let dim_suffix = match dim {
                    crate::ImageDimension::D1 => "_1d",
                    crate::ImageDimension::D2 => "_2d",
                    crate::ImageDimension::D3 => "_3d",
                    crate::ImageDimension::Cube => "_cube",
                };
                let array_suffix = if arrayed { "_array" } else { "" };

                let class_suffix = match class {
                    crate::ImageClass::Sampled { multi: true, .. } => "_multisampled",
                    crate::ImageClass::Depth => "_depth",
                    _ => "",
                };

                let type_in_brackets = match class {
                    crate::ImageClass::Sampled { kind, .. } => {
                        // Note: The only valid widths are 4 bytes wide.
                        // The lexer has already verified this, so we can safely assume it here.
                        // https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                        let element_type = kind.to_wgsl(4);
                        format!("<{}>", element_type)
                    }
                    crate::ImageClass::Depth => String::new(),
                    crate::ImageClass::Storage(format) => {
                        format!("<{}>", format.to_wgsl())
                    }
                };

                format!(
                    "texture{}{}{}{}",
                    class_suffix, dim_suffix, array_suffix, type_in_brackets
                )
            }
            crate::TypeInner::Sampler { .. } => "sampler".to_string(),
        }
    }
}

mod type_inner_tests {
    #[test]
    fn to_wgsl() {
        let mut types = crate::Arena::new();
        let mut constants = crate::Arena::new();
        let c = constants.append(crate::Constant {
            name: Some("C".to_string()),
            specialization: None,
            inner: crate::ConstantInner::Scalar {
                width: 4,
                value: crate::ScalarValue::Uint(32),
            },
        });

        let mytype1 = types.append(crate::Type {
            name: Some("MyType1".to_string()),
            inner: crate::TypeInner::Struct {
                top_level: true,
                members: vec![],
                span: 0,
            },
        });
        let mytype2 = types.append(crate::Type {
            name: Some("MyType2".to_string()),
            inner: crate::TypeInner::Struct {
                top_level: true,
                members: vec![],
                span: 0,
            },
        });

        let array = crate::TypeInner::Array {
            base: mytype1,
            stride: 4,
            size: crate::ArraySize::Constant(c),
        };
        assert_eq!(array.to_wgsl(&types, &constants), "MyType1[C]");

        let mat = crate::TypeInner::Matrix {
            rows: crate::VectorSize::Quad,
            columns: crate::VectorSize::Bi,
            width: 8,
        };
        assert_eq!(mat.to_wgsl(&types, &constants), "mat2x4<f64>");

        let ptr = crate::TypeInner::Pointer {
            base: mytype2,
            class: crate::StorageClass::Storage,
        };
        assert_eq!(ptr.to_wgsl(&types, &constants), "*MyType2");

        let img1 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Sampled {
                kind: crate::ScalarKind::Float,
                multi: true,
            },
        };
        assert_eq!(
            img1.to_wgsl(&types, &constants),
            "texture_multisampled_2d<f32>"
        );

        let img2 = crate::TypeInner::Image {
            dim: crate::ImageDimension::Cube,
            arrayed: true,
            class: crate::ImageClass::Depth,
        };
        assert_eq!(img2.to_wgsl(&types, &constants), "texture_depth_cube_array");
    }
}

impl crate::ScalarKind {
    /// Format a scalar kind+width as a type is written in wgsl.
    ///
    /// Examples: `f32`, `u64`, `bool`.
    fn to_wgsl(self, width: u8) -> String {
        let prefix = match self {
            crate::ScalarKind::Sint => "i",
            crate::ScalarKind::Uint => "u",
            crate::ScalarKind::Float => "f",
            crate::ScalarKind::Bool => return "bool".to_string(),
        };
        format!("{}{}", prefix, width * 8)
    }
}

trait StringValueLookup<'a> {
    type Value;
    fn lookup(&self, key: &'a str, span: Span) -> Result<Self::Value, Error<'a>>;
}
impl<'a> StringValueLookup<'a> for FastHashMap<&'a str, Handle<crate::Expression>> {
    type Value = Handle<crate::Expression>;
    fn lookup(&self, key: &'a str, span: Span) -> Result<Self::Value, Error<'a>> {
        self.get(key).cloned().ok_or(Error::UnknownIdent(span, key))
    }
}

struct StatementContext<'input, 'temp, 'out> {
    lookup_ident: &'temp mut FastHashMap<&'input str, Handle<crate::Expression>>,
    typifier: &'temp mut super::Typifier,
    variables: &'out mut Arena<crate::LocalVariable>,
    expressions: &'out mut Arena<crate::Expression>,
    named_expressions: &'out mut FastHashMap<Handle<crate::Expression>, String>,
    types: &'out mut Arena<crate::Type>,
    constants: &'out mut Arena<crate::Constant>,
    global_vars: &'out Arena<crate::GlobalVariable>,
    functions: &'out Arena<crate::Function>,
    arguments: &'out [crate::FunctionArgument],
}

impl<'a, 'temp> StatementContext<'a, 'temp, '_> {
    fn reborrow(&mut self) -> StatementContext<'a, '_, '_> {
        StatementContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            variables: self.variables,
            expressions: self.expressions,
            named_expressions: self.named_expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            functions: self.functions,
            arguments: self.arguments,
        }
    }

    fn as_expression<'t>(
        &'t mut self,
        block: &'t mut crate::Block,
        emitter: &'t mut super::Emitter,
    ) -> ExpressionContext<'a, 't, '_>
    where
        'temp: 't,
    {
        ExpressionContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            expressions: self.expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.variables,
            functions: self.functions,
            arguments: self.arguments,
            block,
            emitter,
        }
    }
}

struct SamplingContext {
    image: Handle<crate::Expression>,
    arrayed: bool,
}

struct ExpressionContext<'input, 'temp, 'out> {
    lookup_ident: &'temp FastHashMap<&'input str, Handle<crate::Expression>>,
    typifier: &'temp mut super::Typifier,
    expressions: &'out mut Arena<crate::Expression>,
    types: &'out mut Arena<crate::Type>,
    constants: &'out mut Arena<crate::Constant>,
    global_vars: &'out Arena<crate::GlobalVariable>,
    local_vars: &'out Arena<crate::LocalVariable>,
    arguments: &'out [crate::FunctionArgument],
    functions: &'out Arena<crate::Function>,
    block: &'temp mut crate::Block,
    emitter: &'temp mut super::Emitter,
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            lookup_ident: self.lookup_ident,
            typifier: self.typifier,
            expressions: self.expressions,
            types: self.types,
            constants: self.constants,
            global_vars: self.global_vars,
            local_vars: self.local_vars,
            functions: self.functions,
            arguments: self.arguments,
            block: self.block,
            emitter: self.emitter,
        }
    }

    fn resolve_type(
        &mut self,
        handle: Handle<crate::Expression>,
    ) -> Result<&crate::TypeInner, Error<'a>> {
        let resolve_ctx = ResolveContext {
            constants: self.constants,
            types: self.types,
            global_vars: self.global_vars,
            local_vars: self.local_vars,
            functions: self.functions,
            arguments: self.arguments,
        };
        match self.typifier.grow(handle, self.expressions, &resolve_ctx) {
            Err(e) => Err(Error::InvalidResolve(e)),
            Ok(()) => Ok(self.typifier.get(handle, self.types)),
        }
    }

    fn prepare_sampling(
        &mut self,
        image_name: &'a str,
        span: Span,
    ) -> Result<SamplingContext, Error<'a>> {
        let image = self.lookup_ident.lookup(image_name, span.clone())?;
        Ok(SamplingContext {
            image,
            arrayed: match *self.resolve_type(image)? {
                crate::TypeInner::Image { arrayed, .. } => arrayed,
                _ => return Err(Error::BadTexture(span)),
            },
        })
    }

    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            ExpressionContext<'a, '_, '_>,
        ) -> Result<Handle<crate::Expression>, Error<'a>>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let mut left = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            let right = parser(lexer, self.reborrow())?;
            left = self
                .expressions
                .append(crate::Expression::Binary { op, left, right });
        }
        Ok(left)
    }

    fn parse_binary_splat_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            ExpressionContext<'a, '_, '_>,
        ) -> Result<Handle<crate::Expression>, Error<'a>>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let mut left = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            let mut right = parser(lexer, self.reborrow())?;
            // insert splats, if needed by the non-'*' operations
            if op != crate::BinaryOperator::Multiply {
                let left_size = match *self.resolve_type(left)? {
                    crate::TypeInner::Vector { size, .. } => Some(size),
                    _ => None,
                };
                match (left_size, self.resolve_type(right)?) {
                    (Some(size), &crate::TypeInner::Scalar { .. }) => {
                        right = self
                            .expressions
                            .append(crate::Expression::Splat { size, value: right });
                    }
                    (None, &crate::TypeInner::Vector { size, .. }) => {
                        left = self
                            .expressions
                            .append(crate::Expression::Splat { size, value: left });
                    }
                    _ => {}
                }
            }
            left = self
                .expressions
                .append(crate::Expression::Binary { op, left, right });
        }
        Ok(left)
    }
}

enum Composition {
    Single(u32),
    Multi(crate::VectorSize, [crate::SwizzleComponent; 4]),
}

impl Composition {
    //TODO: could be `const fn` once MSRV allows
    fn letter_component(letter: char) -> Option<crate::SwizzleComponent> {
        use crate::SwizzleComponent as Sc;
        match letter {
            'x' | 'r' => Some(Sc::X),
            'y' | 'g' => Some(Sc::Y),
            'z' | 'b' => Some(Sc::Z),
            'w' | 'a' => Some(Sc::W),
            _ => None,
        }
    }

    fn extract_impl(name: &str, name_span: Span) -> Result<u32, Error> {
        let ch = name
            .chars()
            .next()
            .ok_or_else(|| Error::BadAccessor(name_span.clone()))?;
        match Self::letter_component(ch) {
            Some(sc) => Ok(sc as u32),
            None => Err(Error::BadAccessor(name_span)),
        }
    }

    fn extract(
        base: Handle<crate::Expression>,
        name: &str,
        name_span: Span,
    ) -> Result<crate::Expression, Error> {
        Self::extract_impl(name, name_span)
            .map(|index| crate::Expression::AccessIndex { base, index })
    }

    fn make(name: &str, name_span: Span) -> Result<Self, Error> {
        if name.len() > 1 {
            let mut components = [crate::SwizzleComponent::X; 4];
            for (comp, ch) in components.iter_mut().zip(name.chars()) {
                *comp = Self::letter_component(ch)
                    .ok_or_else(|| Error::BadAccessor(name_span.clone()))?;
            }

            let size = match name.len() {
                2 => crate::VectorSize::Bi,
                3 => crate::VectorSize::Tri,
                4 => crate::VectorSize::Quad,
                _ => return Err(Error::BadAccessor(name_span)),
            };
            Ok(Composition::Multi(size, components))
        } else {
            Self::extract_impl(name, name_span).map(Composition::Single)
        }
    }
}

#[derive(Default)]
struct TypeAttributes {
    stride: Option<NonZeroU32>,
    access: crate::StorageAccess,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scope {
    Attribute,
    ImportDecl,
    VariableDecl,
    TypeDecl,
    FunctionDecl,
    Block,
    Statement,
    ConstantExpr,
    PrimaryExpr,
    SingularExpr,
    GeneralExpr,
}

type LocalFunctionCall = (Handle<crate::Function>, Vec<Handle<crate::Expression>>);

#[derive(Default)]
struct BindingParser {
    location: Option<u32>,
    built_in: Option<crate::BuiltIn>,
    interpolation: Option<crate::Interpolation>,
    sampling: Option<crate::Sampling>,
}

impl BindingParser {
    fn parse<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
    ) -> Result<(), Error<'a>> {
        match name {
            "location" => {
                lexer.expect(Token::Paren('('))?;
                self.location = Some(lexer.next_uint_literal()?);
                lexer.expect(Token::Paren(')'))?;
            }
            "builtin" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.built_in = Some(conv::map_built_in(raw, span)?);
                lexer.expect(Token::Paren(')'))?;
            }
            "interpolate" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.interpolation = Some(conv::map_interpolation(raw, span)?);
                if lexer.skip(Token::Separator(',')) {
                    let (raw, span) = lexer.next_ident_with_span()?;
                    self.sampling = Some(conv::map_sampling(raw, span)?);
                }
                lexer.expect(Token::Paren(')'))?;
            }
            _ => return Err(Error::UnknownAttribute(name_span)),
        }
        Ok(())
    }

    fn finish<'a>(self) -> Result<Option<crate::Binding>, Error<'a>> {
        match (
            self.location,
            self.built_in,
            self.interpolation,
            self.sampling,
        ) {
            (None, None, None, None) => Ok(None),
            (Some(location), None, interpolation, sampling) => {
                // Before handing over the completed `Module`, we call
                // `apply_common_default_interpolation` to ensure that the interpolation and
                // sampling have been explicitly specified on all vertex shader output and fragment
                // shader input user bindings, so leaving them potentially `None` here is fine.
                Ok(Some(crate::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                }))
            }
            (None, Some(bi), None, None) => Ok(Some(crate::Binding::BuiltIn(bi))),
            (location, built_in, interpolation, sampling) => Err(Error::InconsistentBinding(
                location,
                built_in,
                interpolation,
                sampling,
            )),
        }
    }
}

struct ParsedVariable<'a> {
    name: &'a str,
    class: Option<crate::StorageClass>,
    ty: Handle<crate::Type>,
    access: crate::StorageAccess,
    init: Option<Handle<crate::Constant>>,
}

#[derive(Clone, Debug)]
pub struct ParseError {
    message: String,
    labels: Vec<(Span, Cow<'static, str>)>,
    notes: Vec<String>,
}

impl ParseError {
    fn diagnostic(&self) -> Diagnostic<()> {
        let diagnostic = Diagnostic::error()
            .with_message(self.message.to_string())
            .with_labels(
                self.labels
                    .iter()
                    .map(|label| {
                        Label::primary((), label.0.clone()).with_message(label.1.to_string())
                    })
                    .collect(),
            )
            .with_notes(
                self.notes
                    .iter()
                    .map(|note| format!("note: {}", note))
                    .collect(),
            );
        diagnostic
    }

    /// Emits a summary of the error to standard error stream.
    pub fn emit_to_stderr(&self, source: &str) {
        let files = SimpleFile::new("wgsl", source);
        let config = codespan_reporting::term::Config::default();
        let writer = StandardStream::stderr(ColorChoice::Always);
        term::emit(&mut writer.lock(), &config, &files, &self.diagnostic())
            .expect("cannot write error");
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string(&self, source: &str) -> String {
        let files = SimpleFile::new("wgsl", source);
        let config = codespan_reporting::term::Config::default();
        let mut writer = StringErrorBuffer::new();
        term::emit(&mut writer, &config, &files, &self.diagnostic()).expect("cannot write error");
        writer.into_string()
    }

    /// Returns the 1-based line number and column of the first label in the
    /// error message.
    pub fn location(&self, source: &str) -> (usize, usize) {
        let files = SimpleFile::new("wgsl", source);
        match self.labels.get(0) {
            Some(label) => {
                let location = files
                    .location((), label.0.start)
                    .expect("invalid span location");
                (location.line_number, location.column_number)
            }
            None => (1, 1),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub struct Parser {
    scopes: Vec<Scope>,
    lookup_type: FastHashMap<String, Handle<crate::Type>>,
    layouter: Layouter,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            lookup_type: FastHashMap::default(),
            layouter: Default::default(),
        }
    }

    fn get_constant_inner<'a>(
        word: &'a str,
        ty: char,
        width: &'a str,
        token: TokenSpan<'a>,
    ) -> Result<ConstantInner, Error<'a>> {
        let span = token.1;
        let value = match ty {
            'i' => word
                .parse()
                .map(crate::ScalarValue::Sint)
                .map_err(|e| Error::BadI32(span.clone(), e))?,
            'u' => word
                .parse()
                .map(crate::ScalarValue::Uint)
                .map_err(|e| Error::BadU32(span.clone(), e))?,
            'f' => word
                .parse()
                .map(crate::ScalarValue::Float)
                .map_err(|e| Error::BadFloat(span.clone(), e))?,
            _ => unreachable!(),
        };
        Ok(crate::ConstantInner::Scalar {
            value,
            width: if width.is_empty() {
                4
            } else {
                match width.parse::<crate::Bytes>() {
                    Ok(bits) if (bits % 8) == 0 => Ok(bits / 8),
                    _ => Err(Error::BadScalarWidth(span, width)),
                }?
            },
        })
    }

    fn parse_local_function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<LocalFunctionCall>, Error<'a>> {
        let fun_handle = match ctx.functions.iter().find(|&(_, fun)| match fun.name {
            Some(ref string) => string == name,
            None => false,
        }) {
            Some((fun_handle, _)) => fun_handle,
            None => return Ok(None),
        };

        let count = ctx.functions[fun_handle].arguments.len();
        let mut arguments = Vec::with_capacity(count);
        lexer.open_arguments()?;
        while arguments.len() != count {
            if !arguments.is_empty() {
                lexer.expect(Token::Separator(','))?;
            }
            let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
            arguments.push(arg);
        }
        lexer.close_arguments()?;
        Ok(Some((fun_handle, arguments)))
    }

    fn parse_function_call_inner<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<Handle<crate::Expression>>, Error<'a>> {
        let expr = if let Some(fun) = conv::map_relational_fun(name) {
            lexer.open_arguments()?;
            let argument = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::Relational { fun, argument }
        } else if let Some(axis) = conv::map_derivative_axis(name) {
            lexer.open_arguments()?;
            let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::Derivative { axis, expr }
        } else if let Some(fun) = conv::map_standard_fun(name) {
            lexer.open_arguments()?;
            let arg_count = fun.argument_count();
            let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
            let arg1 = if arg_count > 1 {
                lexer.expect(Token::Separator(','))?;
                Some(self.parse_general_expression(lexer, ctx.reborrow())?)
            } else {
                None
            };
            let arg2 = if arg_count > 2 {
                lexer.expect(Token::Separator(','))?;
                Some(self.parse_general_expression(lexer, ctx.reborrow())?)
            } else {
                None
            };
            lexer.close_arguments()?;
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            }
        } else if name == "select" {
            lexer.open_arguments()?;
            let accept = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Separator(','))?;
            let reject = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.expect(Token::Separator(','))?;
            let condition = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::Select {
                condition,
                accept,
                reject,
            }
        } else if name == "arrayLength" {
            lexer.open_arguments()?;
            let array = self.parse_singular_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::ArrayLength(array)
        } else {
            // texture sampling
            match name {
                "textureSample" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Auto,
                        depth_ref: None,
                    }
                }
                "textureSampleLevel" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let level = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Exact(level),
                        depth_ref: None,
                    }
                }
                "textureSampleBias" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let bias = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Bias(bias),
                        depth_ref: None,
                    }
                }
                "textureSampleGrad" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let x = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let y = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Gradient { x, y },
                        depth_ref: None,
                    }
                }
                "textureSampleCompare" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let reference = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Auto,
                        depth_ref: Some(reference),
                    }
                }
                "textureSampleCompareLevel" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let (sampler_name, sampler_span) = lexer.next_ident_with_span()?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image_name, image_span)?;
                    let array_index = if sc.arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.expect(Token::Separator(','))?;
                    let reference = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let offset = if lexer.skip(Token::Separator(',')) {
                        Some(self.parse_const_expression(lexer, ctx.types, ctx.constants)?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageSample {
                        image: sc.image,
                        sampler: ctx.lookup_ident.lookup(sampler_name, sampler_span)?,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Zero,
                        depth_ref: Some(reference),
                    }
                }
                "textureLoad" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    let image = ctx.lookup_ident.lookup(image_name, image_span.clone())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let (class, arrayed) = match *ctx.resolve_type(image)? {
                        crate::TypeInner::Image { class, arrayed, .. } => (class, arrayed),
                        _ => return Err(Error::BadTexture(image_span)),
                    };
                    let array_index = if arrayed {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    let index = match class {
                        crate::ImageClass::Storage(_) => None,
                        // it's the MSAA index for multi-sampled, and LOD for the others
                        crate::ImageClass::Sampled { .. } | crate::ImageClass::Depth => {
                            lexer.expect(Token::Separator(','))?;
                            Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                        }
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageLoad {
                        image,
                        coordinate,
                        array_index,
                        index,
                    }
                }
                "textureDimensions" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    let image = ctx.lookup_ident.lookup(image_name, image_span)?;
                    let level = if lexer.skip(Token::Separator(',')) {
                        let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                        Some(expr)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::Size { level },
                    }
                }
                "textureNumLevels" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    let image = ctx.lookup_ident.lookup(image_name, image_span)?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLevels,
                    }
                }
                "textureNumLayers" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    let image = ctx.lookup_ident.lookup(image_name, image_span)?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLayers,
                    }
                }
                "textureNumSamples" => {
                    lexer.open_arguments()?;
                    let (image_name, image_span) = lexer.next_ident_with_span()?;
                    let image = ctx.lookup_ident.lookup(image_name, image_span)?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumSamples,
                    }
                }
                // other
                _ => {
                    let handle =
                        match self.parse_local_function_call(lexer, name, ctx.reborrow())? {
                            Some((function, arguments)) => {
                                ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                                let result =
                                    Some(ctx.expressions.append(crate::Expression::Call(function)));
                                ctx.block.push(crate::Statement::Call {
                                    function,
                                    arguments,
                                    result,
                                });
                                // restart the emitter
                                ctx.emitter.start(ctx.expressions);
                                result
                            }
                            None => None,
                        };
                    return Ok(handle);
                }
            }
        };
        Ok(Some(ctx.expressions.append(expr)))
    }

    fn parse_construction<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<Handle<crate::Expression>>, Error<'a>> {
        let ty_resolution = match self.lookup_type.get(type_name) {
            Some(&handle) => TypeResolution::Handle(handle),
            None => match self.parse_type_decl_impl(
                lexer,
                TypeAttributes::default(),
                type_name,
                ctx.types,
                ctx.constants,
            )? {
                Some(inner) => TypeResolution::Value(inner),
                None => return Ok(None),
            },
        };

        let mut components = Vec::new();
        let (last_component, arguments_span) = lexer.capture_span(|lexer| {
            lexer.open_arguments()?;
            let mut last_component = self.parse_general_expression(lexer, ctx.reborrow())?;

            while lexer.next_argument()? {
                components.push(last_component);
                last_component = self.parse_general_expression(lexer, ctx.reborrow())?;
            }

            Ok(last_component)
        })?;

        let expr = if components.is_empty()
            && ty_resolution.inner_with(ctx.types).scalar_kind().is_some()
        {
            // We can't use the `TypeInner` returned by this because
            // `resolve_type` borrows context mutably.
            // Use it to insert into the right maps,
            // and then grab it again immutably.
            ctx.resolve_type(last_component)?;

            match (
                ty_resolution.inner_with(ctx.types),
                ctx.typifier.get(last_component, ctx.types),
            ) {
                (&crate::TypeInner::Vector { size, .. }, &crate::TypeInner::Scalar { .. }) => {
                    crate::Expression::Splat {
                        size,
                        value: last_component,
                    }
                }
                (
                    &crate::TypeInner::Scalar { kind, width, .. },
                    &crate::TypeInner::Scalar { .. },
                )
                | (
                    &crate::TypeInner::Vector { kind, width, .. },
                    &crate::TypeInner::Vector { .. },
                ) => crate::Expression::As {
                    expr: last_component,
                    kind,
                    convert: Some(width),
                },
                (&crate::TypeInner::Matrix { width, .. }, &crate::TypeInner::Matrix { .. }) => {
                    crate::Expression::As {
                        expr: last_component,
                        kind: crate::ScalarKind::Float,
                        convert: Some(width),
                    }
                }
                (to_type, from_type) => {
                    return Err(Error::BadTypeCast {
                        span: arguments_span,
                        from_type: from_type.to_wgsl(ctx.types, ctx.constants),
                        to_type: to_type.to_wgsl(ctx.types, ctx.constants),
                    });
                }
            }
        } else {
            let ty = match ty_resolution {
                TypeResolution::Handle(handle) => handle,
                TypeResolution::Value(inner) => {
                    ctx.types.fetch_or_append(crate::Type { name: None, inner })
                }
            };
            components.push(last_component);
            crate::Expression::Compose { ty, components }
        };

        Ok(Some(ctx.expressions.append(expr)))
    }

    fn parse_const_expression_impl<'a>(
        &mut self,
        first_token_span: TokenSpan<'a>,
        lexer: &mut Lexer<'a>,
        register_name: Option<&'a str>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.scopes.push(Scope::ConstantExpr);
        let inner = match first_token_span {
            (Token::Word("true"), _) => crate::ConstantInner::boolean(true),
            (Token::Word("false"), _) => crate::ConstantInner::boolean(false),
            (Token::Number { value, ty, width }, _) => {
                Self::get_constant_inner(value, ty, width, first_token_span)?
            }
            (Token::Word(name), name_span) => {
                // look for an existing constant first
                for (handle, var) in const_arena.iter() {
                    match var.name {
                        Some(ref string) if string == name => {
                            self.scopes.pop();
                            return Ok(handle);
                        }
                        _ => {}
                    }
                }
                let composite_ty = self.parse_type_decl_name(
                    lexer,
                    name,
                    name_span,
                    None,
                    TypeAttributes::default(),
                    type_arena,
                    const_arena,
                )?;

                lexer.open_arguments()?;
                //Note: this expects at least one argument
                let mut components = Vec::new();
                while components.is_empty() || lexer.next_argument()? {
                    let component = self.parse_const_expression(lexer, type_arena, const_arena)?;
                    components.push(component);
                }
                crate::ConstantInner::Composite {
                    ty: composite_ty,
                    components,
                }
            }
            other => return Err(Error::Unexpected(other, ExpectedToken::Constant)),
        };

        let handle = if let Some(name) = register_name {
            const_arena.append(crate::Constant {
                name: Some(name.to_string()),
                specialization: None,
                inner,
            })
        } else {
            const_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner,
            })
        };

        self.scopes.pop();
        Ok(handle)
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.parse_const_expression_impl(lexer.next(), lexer, None, type_arena, const_arena)
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::PrimaryExpr);
        let handle = match lexer.next() {
            (Token::Paren('('), _) => {
                let expr = self.parse_general_expression(lexer, ctx)?;
                lexer.expect(Token::Paren(')'))?;
                expr
            }
            token @ (Token::Word("true"), _)
            | token @ (Token::Word("false"), _)
            | token @ (Token::Number { .. }, _) => {
                let const_handle =
                    self.parse_const_expression_impl(token, lexer, None, ctx.types, ctx.constants)?;
                // pause the emitter while generating this expression, since it's pre-emitted
                ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                let expr = ctx
                    .expressions
                    .append(crate::Expression::Constant(const_handle));
                ctx.emitter.start(ctx.expressions);
                expr
            }
            (Token::Word(word), span) => {
                if let Some(&expr) = ctx.lookup_ident.get(word) {
                    expr
                } else if let Some(expr) =
                    self.parse_function_call_inner(lexer, word, ctx.reborrow())?
                {
                    //TODO: resolve the duplicate call in `parse_singular_expression`
                    expr
                } else if let Some(expr) = self.parse_construction(lexer, word, ctx.reborrow())? {
                    expr
                } else {
                    return Err(Error::UnknownIdent(span, word));
                }
            }
            other => return Err(Error::Unexpected(other, ExpectedToken::PrimaryExpression)),
        };
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_postfix<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
        mut handle: Handle<crate::Expression>,
        allow_deref: bool,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let mut needs_deref = match ctx.expressions[handle] {
            crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                allow_deref
            }
            _ => false,
        };
        loop {
            // insert the E::Load when we reach a value
            if needs_deref {
                let now = match *ctx.resolve_type(handle)? {
                    crate::TypeInner::Pointer { base, class: _ } => match ctx.types[base].inner {
                        crate::TypeInner::Scalar { .. } | crate::TypeInner::Vector { .. } => true,
                        _ => false,
                    },
                    crate::TypeInner::ValuePointer { .. } => true,
                    _ => false,
                };
                if now {
                    let expression = crate::Expression::Load { pointer: handle };
                    handle = ctx.expressions.append(expression);
                    needs_deref = false;
                }
            }

            let expression = match lexer.peek().0 {
                Token::Separator('.') => {
                    let _ = lexer.next();
                    let (name, name_span) = lexer.next_ident_with_span()?;
                    match *ctx.resolve_type(handle)? {
                        crate::TypeInner::Struct { ref members, .. } => {
                            let index = members
                                .iter()
                                .position(|m| m.name.as_deref() == Some(name))
                                .ok_or(Error::BadAccessor(name_span))?
                                as u32;
                            crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            }
                        }
                        crate::TypeInner::Vector { .. } | crate::TypeInner::Matrix { .. } => {
                            match Composition::make(name, name_span)? {
                                Composition::Multi(dst_size, pattern) => {
                                    crate::Expression::Swizzle {
                                        size: dst_size,
                                        vector: handle,
                                        pattern,
                                    }
                                }
                                Composition::Single(index) => crate::Expression::AccessIndex {
                                    base: handle,
                                    index,
                                },
                            }
                        }
                        crate::TypeInner::ValuePointer { .. } => {
                            Composition::extract(handle, name, name_span)?
                        }
                        crate::TypeInner::Pointer { base, class: _ } => match ctx.types[base].inner
                        {
                            crate::TypeInner::Struct { ref members, .. } => {
                                let index = members
                                    .iter()
                                    .position(|m| m.name.as_deref() == Some(name))
                                    .ok_or(Error::BadAccessor(name_span))?
                                    as u32;
                                crate::Expression::AccessIndex {
                                    base: handle,
                                    index,
                                }
                            }
                            _ => Composition::extract(handle, name, name_span)?,
                        },
                        _ => return Err(Error::BadAccessor(name_span)),
                    }
                }
                Token::Paren('[') => {
                    let (_, open_brace_span) = lexer.next();
                    let index = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let close_brace_span = lexer.expect_span(Token::Paren(']'))?;

                    if let crate::Expression::Constant(constant) = ctx.expressions[index] {
                        let expr_span = open_brace_span.end..close_brace_span.start;

                        let index = match ctx.constants[constant].inner {
                            ConstantInner::Scalar {
                                value: ScalarValue::Uint(int),
                                ..
                            } => u32::try_from(int).map_err(|_| Error::BadU32Constant(expr_span)),
                            ConstantInner::Scalar {
                                value: ScalarValue::Sint(int),
                                ..
                            } => u32::try_from(int).map_err(|_| Error::BadU32Constant(expr_span)),
                            _ => Err(Error::BadU32Constant(expr_span)),
                        }?;

                        crate::Expression::AccessIndex {
                            base: handle,
                            index,
                        }
                    } else {
                        crate::Expression::Access {
                            base: handle,
                            index,
                        }
                    }
                }
                _ => {
                    // after we reached for the value, load it
                    return Ok(if needs_deref {
                        let expression = crate::Expression::Load { pointer: handle };
                        ctx.expressions.append(expression)
                    } else {
                        handle
                    });
                }
            };

            handle = ctx.expressions.append(expression);
        }
    }

    fn parse_singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::SingularExpr);
        //TODO: refactor this to avoid backing up
        let backup = lexer.clone();
        let (allow_deref, handle) = match lexer.next().0 {
            Token::Operation('-') => {
                let expr = crate::Expression::Unary {
                    op: crate::UnaryOperator::Negate,
                    expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
                };
                (true, ctx.expressions.append(expr))
            }
            Token::Operation('!') | Token::Operation('~') => {
                let expr = crate::Expression::Unary {
                    op: crate::UnaryOperator::Not,
                    expr: self.parse_singular_expression(lexer, ctx.reborrow())?,
                };
                (true, ctx.expressions.append(expr))
            }
            Token::Operation('&') => {
                let handle = self.parse_primary_expression(lexer, ctx.reborrow())?;
                (false, handle)
            }
            Token::Word(word) => {
                let handle = match self.parse_function_call_inner(lexer, word, ctx.reborrow())? {
                    Some(handle) => handle,
                    None => {
                        *lexer = backup;
                        self.parse_primary_expression(lexer, ctx.reborrow())?
                    }
                };
                (true, handle)
            }
            _ => {
                *lexer = backup;
                let handle = self.parse_primary_expression(lexer, ctx.reborrow())?;
                (true, handle)
            }
        };

        let post_handle = self.parse_postfix(lexer, ctx, handle, allow_deref)?;
        self.scopes.pop();
        Ok(post_handle)
    }

    fn parse_equality_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        // equality_expression
        context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('=') => Some(crate::BinaryOperator::Equal),
                Token::LogicalOperation('!') => Some(crate::BinaryOperator::NotEqual),
                _ => None,
            },
            // relational_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::Paren('<') => Some(crate::BinaryOperator::Less),
                        Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                        Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                        Token::LogicalOperation('>') => Some(crate::BinaryOperator::GreaterEqual),
                        _ => None,
                    },
                    // shift_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::ShiftOperation('<') => {
                                    Some(crate::BinaryOperator::ShiftLeft)
                                }
                                Token::ShiftOperation('>') => {
                                    Some(crate::BinaryOperator::ShiftRight)
                                }
                                _ => None,
                            },
                            // additive_expression
                            |lexer, mut context| {
                                context.parse_binary_splat_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('+') => Some(crate::BinaryOperator::Add),
                                        Token::Operation('-') => {
                                            Some(crate::BinaryOperator::Subtract)
                                        }
                                        _ => None,
                                    },
                                    // multiplicative_expression
                                    |lexer, mut context| {
                                        context.parse_binary_splat_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('*') => {
                                                    Some(crate::BinaryOperator::Multiply)
                                                }
                                                Token::Operation('/') => {
                                                    Some(crate::BinaryOperator::Divide)
                                                }
                                                Token::Operation('%') => {
                                                    Some(crate::BinaryOperator::Modulo)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_singular_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )
    }

    fn parse_general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        self.scopes.push(Scope::GeneralExpr);
        // logical_or_expression
        let handle = context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('|') => Some(crate::BinaryOperator::LogicalOr),
                _ => None,
            },
            // logical_and_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::LogicalOperation('&') => Some(crate::BinaryOperator::LogicalAnd),
                        _ => None,
                    },
                    // inclusive_or_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('|') => Some(crate::BinaryOperator::InclusiveOr),
                                _ => None,
                            },
                            // exclusive_or_expression
                            |lexer, mut context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('^') => {
                                            Some(crate::BinaryOperator::ExclusiveOr)
                                        }
                                        _ => None,
                                    },
                                    // and_expression
                                    |lexer, mut context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('&') => {
                                                    Some(crate::BinaryOperator::And)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_equality_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )?;
        self.scopes.pop();
        Ok(handle)
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(&'a str, Handle<crate::Type>, crate::StorageAccess), Error<'a>> {
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let (ty, access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
        Ok((name, ty, access))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<ParsedVariable<'a>, Error<'a>> {
        self.scopes.push(Scope::VariableDecl);
        let mut class = None;
        if lexer.skip(Token::Paren('<')) {
            let (class_str, span) = lexer.next_ident_with_span()?;
            class = Some(conv::map_storage_class(class_str, span)?);
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let (ty, access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;

        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.parse_const_expression(lexer, type_arena, const_arena)?;
            Some(handle)
        } else {
            None
        };
        lexer.expect(Token::Separator(';'))?;
        self.scopes.pop();
        Ok(ParsedVariable {
            name,
            class,
            ty,
            access,
            init,
        })
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(Vec<crate::StructMember>, u32), Error<'a>> {
        let mut offset = 0;
        let mut alignment = Alignment::new(1).unwrap();
        let mut members = Vec::new();

        lexer.expect(Token::Paren('{'))?;
        loop {
            let (mut size, mut align) = (None, None);
            let mut bind_parser = BindingParser::default();
            if lexer.skip(Token::DoubleParen('[')) {
                self.scopes.push(Scope::Attribute);
                let mut ready = true;
                loop {
                    match lexer.next() {
                        (Token::DoubleParen(']'), _) => {
                            break;
                        }
                        (Token::Separator(','), _) if !ready => {
                            ready = true;
                        }
                        (Token::Word(word), word_span) if ready => {
                            match word {
                                "size" => {
                                    lexer.expect(Token::Paren('('))?;
                                    let (value, span) =
                                        lexer.capture_span(Lexer::next_uint_literal)?;
                                    lexer.expect(Token::Paren(')'))?;
                                    size = Some(
                                        NonZeroU32::new(value)
                                            .ok_or(Error::ZeroSizeOrAlign(span))?,
                                    );
                                }
                                "align" => {
                                    lexer.expect(Token::Paren('('))?;
                                    let (value, span) =
                                        lexer.capture_span(Lexer::next_uint_literal)?;
                                    lexer.expect(Token::Paren(')'))?;
                                    align = Some(
                                        NonZeroU32::new(value)
                                            .ok_or(Error::ZeroSizeOrAlign(span))?,
                                    );
                                }
                                _ => bind_parser.parse(lexer, word, word_span)?,
                            }
                            ready = false;
                        }
                        other if ready => {
                            return Err(Error::Unexpected(other, ExpectedToken::StructAttribute))
                        }
                        other => {
                            return Err(Error::Unexpected(other, ExpectedToken::AttributeSeparator))
                        }
                    }
                }
                self.scopes.pop();
            }

            let name = match lexer.next() {
                (Token::Word(word), _) => word,
                (Token::Paren('}'), _) => {
                    let span = Layouter::round_up(alignment, offset);
                    return Ok((members, span));
                }
                other => return Err(Error::Unexpected(other, ExpectedToken::FieldName)),
            };
            lexer.expect(Token::Separator(':'))?;
            let (ty, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
            lexer.expect(Token::Separator(';'))?;

            self.layouter.update(type_arena, const_arena).unwrap();

            let (range, align) = self.layouter.member_placement(offset, ty, align, size);
            alignment = alignment.max(align);
            offset = range.end;

            members.push(crate::StructMember {
                name: Some(name.to_owned()),
                ty,
                binding: bind_parser.finish()?,
                offset: range.start,
            });
        }
    }

    fn parse_type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        attribute: TypeAttributes,
        word: &'a str,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Option<crate::TypeInner>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(crate::TypeInner::Scalar { kind, width }));
        }
        Ok(Some(match word {
            "vec2" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            "vec3" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            "vec4" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            "mat2x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat2x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat2x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "mat3x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat3x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat3x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "mat4x2" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Bi,
                    width,
                }
            }
            "mat4x3" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    width,
                }
            }
            "mat4x4" => {
                let (_, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    width,
                }
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let (ident, span) = lexer.next_ident_with_span()?;
                let class = conv::map_storage_class(ident, span)?;
                lexer.expect(Token::Separator(','))?;
                let (base, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                lexer.expect_generic_paren('>')?;
                crate::TypeInner::Pointer { base, class }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let (base, _access) = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let const_handle =
                        self.parse_const_expression(lexer, type_arena, const_arena)?;
                    crate::ArraySize::Constant(const_handle)
                } else {
                    crate::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;
                let stride = match attribute.stride {
                    Some(stride) => stride.get(),
                    None => type_arena[base].inner.span(const_arena),
                };

                crate::TypeInner::Array { base, size, stride }
            }
            "sampler" => crate::TypeInner::Sampler { comparison: false },
            "sampler_comparison" => crate::TypeInner::Sampler { comparison: true },
            "texture_1d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_1d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_3d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_multisampled_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_multisampled_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_depth_2d" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_2d_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_cube" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth,
            },
            "texture_depth_cube_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth,
            },
            "texture_storage_1d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_1d_array" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_2d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_2d_array" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage(format),
                }
            }
            "texture_storage_3d" => {
                let format = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage(format),
                }
            }
            _ => return Ok(None),
        }))
    }

    fn check_texture_sample_type(
        kind: crate::ScalarKind,
        width: u8,
        span: Span,
    ) -> Result<(), Error<'static>> {
        use crate::ScalarKind::*;
        // Validate according to https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
        match (kind, width) {
            (Float, 4) | (Sint, 4) | (Uint, 4) => Ok(()),
            _ => Err(Error::BadTextureSampleType { span, kind, width }),
        }
    }

    /// Parse type declaration of a given name and attribute.
    #[allow(clippy::too_many_arguments)]
    fn parse_type_decl_name<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        debug_name: Option<&'a str>,
        attribute: TypeAttributes,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        Ok(match self.lookup_type.get(name) {
            Some(&handle) => handle,
            None => {
                match self.parse_type_decl_impl(lexer, attribute, name, type_arena, const_arena)? {
                    Some(inner) => type_arena.fetch_or_append(crate::Type {
                        name: debug_name.map(|s| s.to_string()),
                        inner,
                    }),
                    None => return Err(Error::UnknownType(name_span)),
                }
            }
        })
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        debug_name: Option<&'a str>,
        type_arena: &mut Arena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(Handle<crate::Type>, crate::StorageAccess), Error<'a>> {
        self.scopes.push(Scope::TypeDecl);
        let mut attribute = TypeAttributes::default();

        if lexer.skip(Token::DoubleParen('[')) {
            self.scopes.push(Scope::Attribute);
            loop {
                match lexer.next() {
                    (Token::Word("access"), _) => {
                        lexer.expect(Token::Paren('('))?;
                        attribute.access = match lexer.next_ident()? {
                            "read" => crate::StorageAccess::LOAD,
                            "write" => crate::StorageAccess::STORE,
                            "read_write" => crate::StorageAccess::all(),
                            other => return Err(Error::UnknownAccess(other)),
                        };
                        lexer.expect(Token::Paren(')'))?;
                    }
                    (Token::Word("stride"), _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (stride, span) = lexer.capture_span(Lexer::next_uint_literal)?;
                        attribute.stride =
                            Some(NonZeroU32::new(stride).ok_or(Error::ZeroStride(span))?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    (Token::DoubleParen(']'), _) => break,
                    other => return Err(Error::Unexpected(other, ExpectedToken::TypeAttribute)),
                }
            }
            self.scopes.pop();
        }

        let storage_access = attribute.access;
        let (name, name_span) = lexer.next_ident_with_span()?;
        let handle = self.parse_type_decl_name(
            lexer,
            name,
            name_span,
            debug_name,
            attribute,
            type_arena,
            const_arena,
        )?;
        self.scopes.pop();
        Ok((handle, storage_access))
    }

    /// Parse a statement that is either an assignment or a function call.
    fn parse_statement_restricted<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ident: &'a str,
        mut context: ExpressionContext<'a, '_, 'out>,
    ) -> Result<(), Error<'a>> {
        context.emitter.start(context.expressions);
        let stmt = match context.lookup_ident.get(ident) {
            Some(&expr) => {
                let left = self.parse_postfix(lexer, context.reborrow(), expr, false)?;
                lexer.expect(Token::Operation('='))?;
                let value = self.parse_general_expression(lexer, context.reborrow())?;
                crate::Statement::Store {
                    pointer: left,
                    value,
                }
            }
            None => {
                let (function, arguments) = self
                    .parse_local_function_call(lexer, ident, context.reborrow())?
                    .ok_or(Error::UnknownLocalFunction(ident))?;
                crate::Statement::Call {
                    function,
                    arguments,
                    result: None,
                }
            }
        };
        context
            .block
            .extend(context.emitter.finish(context.expressions));
        context.block.push(stmt);
        Ok(())
    }

    fn parse_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, 'out>,
        block: &'out mut crate::Block,
        is_uniform_control_flow: bool,
    ) -> Result<(), Error<'a>> {
        let word = match lexer.next() {
            (Token::Separator(';'), _) => return Ok(()),
            (Token::Paren('{'), _) => {
                self.scopes.push(Scope::Block);
                let mut statements = Vec::new();
                while !lexer.skip(Token::Paren('}')) {
                    self.parse_statement(
                        lexer,
                        context.reborrow(),
                        &mut statements,
                        is_uniform_control_flow,
                    )?;
                }
                self.scopes.pop();
                block.push(crate::Statement::Block(statements));
                return Ok(());
            }
            (Token::Word(word), _) => word,
            other => return Err(Error::Unexpected(other, ExpectedToken::Statement)),
        };

        self.scopes.push(Scope::Statement);
        let mut emitter = super::Emitter::default();
        match word {
            "let" => {
                emitter.start(context.expressions);
                let name = lexer.next_ident()?;
                let given_ty = if lexer.skip(Token::Separator(':')) {
                    let (ty, _access) =
                        self.parse_type_decl(lexer, None, context.types, context.constants)?;
                    Some(ty)
                } else {
                    None
                };
                lexer.expect(Token::Operation('='))?;
                let expr_id = self
                    .parse_general_expression(lexer, context.as_expression(block, &mut emitter))?;
                lexer.expect(Token::Separator(';'))?;
                if let Some(ty) = given_ty {
                    // prepare the typifier, but work around mutable borrowing...
                    let _ = context
                        .as_expression(block, &mut emitter)
                        .resolve_type(expr_id)?;
                    let expr_inner = context.typifier.get(expr_id, context.types);
                    let given_inner = &context.types[ty].inner;
                    if given_inner != expr_inner {
                        log::error!(
                            "Given type {:?} doesn't match expected {:?}",
                            given_inner,
                            expr_inner
                        );
                        return Err(Error::LetTypeMismatch(name, ty));
                    }
                }
                block.extend(emitter.finish(context.expressions));
                context.lookup_ident.insert(name, expr_id);
                context
                    .named_expressions
                    .insert(expr_id, String::from(name));
            }
            "var" => {
                enum Init {
                    Empty,
                    Constant(Handle<crate::Constant>),
                    Variable(Handle<crate::Expression>),
                }

                let (name, ty, _access) =
                    self.parse_variable_ident_decl(lexer, context.types, context.constants)?;

                let init = if lexer.skip(Token::Operation('=')) {
                    emitter.start(context.expressions);
                    let value = self.parse_general_expression(
                        lexer,
                        context.as_expression(block, &mut emitter),
                    )?;
                    block.extend(emitter.finish(context.expressions));
                    match context.expressions[value] {
                        crate::Expression::Constant(handle) if is_uniform_control_flow => {
                            Init::Constant(handle)
                        }
                        _ => Init::Variable(value),
                    }
                } else {
                    Init::Empty
                };

                lexer.expect(Token::Separator(';'))?;
                let var_id = context.variables.append(crate::LocalVariable {
                    name: Some(name.to_owned()),
                    ty,
                    init: match init {
                        Init::Constant(value) => Some(value),
                        _ => None,
                    },
                });

                let expr_id = context
                    .expressions
                    .append(crate::Expression::LocalVariable(var_id));
                context.lookup_ident.insert(name, expr_id);

                if let Init::Variable(value) = init {
                    block.push(crate::Statement::Store {
                        pointer: expr_id,
                        value,
                    });
                }
            }
            "return" => {
                let value = if lexer.peek().0 != Token::Separator(';') {
                    emitter.start(context.expressions);
                    let handle = self.parse_general_expression(
                        lexer,
                        context.as_expression(block, &mut emitter),
                    )?;
                    block.extend(emitter.finish(context.expressions));
                    Some(handle)
                } else {
                    None
                };
                lexer.expect(Token::Separator(';'))?;
                block.push(crate::Statement::Return { value });
            }
            "if" => {
                emitter.start(context.expressions);
                lexer.expect(Token::Paren('('))?;
                let condition = self
                    .parse_general_expression(lexer, context.as_expression(block, &mut emitter))?;
                lexer.expect(Token::Paren(')'))?;
                block.extend(emitter.finish(context.expressions));

                let accept = self.parse_block(lexer, context.reborrow(), false)?;
                let mut elsif_stack = Vec::new();
                while lexer.skip(Token::Word("elseif")) {
                    let mut sub_emitter = super::Emitter::default();
                    sub_emitter.start(context.expressions);
                    lexer.expect(Token::Paren('('))?;
                    let other_condition = self.parse_general_expression(
                        lexer,
                        context.as_expression(block, &mut sub_emitter),
                    )?;
                    lexer.expect(Token::Paren(')'))?;
                    let other_emit = sub_emitter.finish(context.expressions);
                    let other_block = self.parse_block(lexer, context.reborrow(), false)?;
                    elsif_stack.push((other_condition, other_emit, other_block));
                }
                let mut reject = if lexer.skip(Token::Word("else")) {
                    self.parse_block(lexer, context.reborrow(), false)?
                } else {
                    Vec::new()
                };
                // reverse-fold the else-if blocks
                //Note: we may consider uplifting this to the IR
                for (other_cond, other_emit, other_block) in elsif_stack.drain(..).rev() {
                    reject = other_emit
                        .into_iter()
                        .chain(iter::once(crate::Statement::If {
                            condition: other_cond,
                            accept: other_block,
                            reject,
                        }))
                        .collect();
                }

                block.push(crate::Statement::If {
                    condition,
                    accept,
                    reject,
                });
            }
            "switch" => {
                emitter.start(context.expressions);
                lexer.expect(Token::Paren('('))?;
                let selector = self
                    .parse_general_expression(lexer, context.as_expression(block, &mut emitter))?;
                lexer.expect(Token::Paren(')'))?;
                block.extend(emitter.finish(context.expressions));
                lexer.expect(Token::Paren('{'))?;
                let mut cases = Vec::new();
                let mut default = Vec::new();

                loop {
                    // cases + default
                    match lexer.next() {
                        (Token::Word("case"), _) => {
                            // parse a list of values
                            let value = loop {
                                let value = lexer.next_sint_literal()?;
                                if lexer.skip(Token::Separator(',')) {
                                    if lexer.skip(Token::Separator(':')) {
                                        break value;
                                    }
                                } else {
                                    lexer.expect(Token::Separator(':'))?;
                                    break value;
                                }
                                cases.push(crate::SwitchCase {
                                    value,
                                    body: Vec::new(),
                                    fall_through: true,
                                });
                            };

                            let mut body = Vec::new();
                            lexer.expect(Token::Paren('{'))?;
                            let fall_through = loop {
                                // default statements
                                if lexer.skip(Token::Word("fallthrough")) {
                                    lexer.expect(Token::Separator(';'))?;
                                    lexer.expect(Token::Paren('}'))?;
                                    break true;
                                }
                                if lexer.skip(Token::Paren('}')) {
                                    break false;
                                }
                                self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
                            };

                            cases.push(crate::SwitchCase {
                                value,
                                body,
                                fall_through,
                            });
                        }
                        (Token::Word("default"), _) => {
                            lexer.expect(Token::Separator(':'))?;
                            default = self.parse_block(lexer, context.reborrow(), false)?;
                        }
                        (Token::Paren('}'), _) => break,
                        other => return Err(Error::Unexpected(other, ExpectedToken::SwitchItem)),
                    }
                }

                block.push(crate::Statement::Switch {
                    selector,
                    cases,
                    default,
                });
            }
            "loop" => {
                let mut body = Vec::new();
                let mut continuing = Vec::new();
                lexer.expect(Token::Paren('{'))?;

                loop {
                    if lexer.skip(Token::Word("continuing")) {
                        continuing = self.parse_block(lexer, context.reborrow(), false)?;
                        lexer.expect(Token::Paren('}'))?;
                        break;
                    }
                    if lexer.skip(Token::Paren('}')) {
                        break;
                    }
                    self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
                }

                block.push(crate::Statement::Loop { body, continuing });
            }
            "for" => {
                lexer.expect(Token::Paren('('))?;
                if !lexer.skip(Token::Separator(';')) {
                    let num_statements = block.len();
                    let (_, span) = lexer.capture_span(|lexer| {
                        self.parse_statement(
                            lexer,
                            context.reborrow(),
                            block,
                            is_uniform_control_flow,
                        )
                    })?;

                    if block.len() != num_statements {
                        match *block.last().unwrap() {
                            crate::Statement::Store { .. } | crate::Statement::Call { .. } => {}
                            _ => return Err(Error::InvalidForInitializer(span)),
                        }
                    }
                };

                let mut body = Vec::new();
                if !lexer.skip(Token::Separator(';')) {
                    emitter.start(context.expressions);
                    let condition = self.parse_general_expression(
                        lexer,
                        context.as_expression(&mut body, &mut emitter),
                    )?;
                    lexer.expect(Token::Separator(';'))?;
                    body.extend(emitter.finish(context.expressions));
                    body.push(crate::Statement::If {
                        condition,
                        accept: Vec::new(),
                        reject: vec![crate::Statement::Break],
                    });
                };

                let mut continuing = Vec::new();
                if let Token::Word(ident) = lexer.peek().0 {
                    // manually parse the next statement here instead of calling parse_statement
                    // because the statement is not terminated with a semicolon
                    let _ = lexer.next();
                    self.parse_statement_restricted(
                        lexer,
                        ident,
                        context.as_expression(&mut continuing, &mut emitter),
                    )?;
                }
                lexer.expect(Token::Paren(')'))?;
                lexer.expect(Token::Paren('{'))?;

                while !lexer.skip(Token::Paren('}')) {
                    self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
                }

                block.push(crate::Statement::Loop { body, continuing });
            }
            "break" => block.push(crate::Statement::Break),
            "continue" => block.push(crate::Statement::Continue),
            "discard" => block.push(crate::Statement::Kill),
            "storageBarrier" => {
                lexer.expect(Token::Paren('('))?;
                lexer.expect(Token::Paren(')'))?;
                block.push(crate::Statement::Barrier(crate::Barrier::STORAGE));
            }
            "workgroupBarrier" => {
                lexer.expect(Token::Paren('('))?;
                lexer.expect(Token::Paren(')'))?;
                block.push(crate::Statement::Barrier(crate::Barrier::WORK_GROUP));
            }
            "textureStore" => {
                emitter.start(context.expressions);
                lexer.open_arguments()?;
                let (image_name, image_span) = lexer.next_ident_with_span()?;
                let image = context
                    .lookup_ident
                    .lookup(image_name, image_span.clone())?;
                lexer.expect(Token::Separator(','))?;
                let mut expr_context = context.as_expression(block, &mut emitter);
                let arrayed = match *expr_context.resolve_type(image)? {
                    crate::TypeInner::Image { arrayed, .. } => arrayed,
                    _ => return Err(Error::BadTexture(image_span)),
                };
                let coordinate = self.parse_general_expression(lexer, expr_context)?;
                let array_index = if arrayed {
                    lexer.expect(Token::Separator(','))?;
                    Some(self.parse_general_expression(
                        lexer,
                        context.as_expression(block, &mut emitter),
                    )?)
                } else {
                    None
                };
                lexer.expect(Token::Separator(','))?;
                let value = self
                    .parse_general_expression(lexer, context.as_expression(block, &mut emitter))?;
                lexer.close_arguments()?;
                block.extend(emitter.finish(context.expressions));
                block.push(crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                });
            }
            // assignment or a function call
            ident => {
                self.parse_statement_restricted(
                    lexer,
                    ident,
                    context.as_expression(block, &mut emitter),
                )?;
                lexer.expect(Token::Separator(';'))?;
            }
        }
        self.scopes.pop();
        Ok(())
    }

    fn parse_block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
        is_uniform_control_flow: bool,
    ) -> Result<Vec<crate::Statement>, Error<'a>> {
        self.scopes.push(Scope::Block);
        lexer.expect(Token::Paren('{'))?;
        let mut block = Vec::new();
        while !lexer.skip(Token::Paren('}')) {
            self.parse_statement(
                lexer,
                context.reborrow(),
                &mut block,
                is_uniform_control_flow,
            )?;
        }
        self.scopes.pop();
        Ok(block)
    }

    fn parse_varying_binding<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
    ) -> Result<Option<crate::Binding>, Error<'a>> {
        if !lexer.skip(Token::DoubleParen('[')) {
            return Ok(None);
        }

        let mut bind_parser = BindingParser::default();
        self.scopes.push(Scope::Attribute);
        loop {
            let (word, span) = lexer.next_ident_with_span()?;
            bind_parser.parse(lexer, word, span)?;
            match lexer.next() {
                (Token::DoubleParen(']'), _) => {
                    break;
                }
                (Token::Separator(','), _) => {}
                other => return Err(Error::Unexpected(other, ExpectedToken::AttributeSeparator)),
            }
        }
        self.scopes.pop();
        bind_parser.finish()
    }

    fn parse_function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &FastHashMap<&'a str, crate::Expression>,
    ) -> Result<(crate::Function, &'a str), Error<'a>> {
        self.scopes.push(Scope::FunctionDecl);
        // read function name
        let mut lookup_ident = FastHashMap::default();
        let fun_name = lexer.next_ident()?;
        // populate initial expressions
        let mut expressions = Arena::new();
        for (&name, expression) in lookup_global_expression.iter() {
            let expr_handle = expressions.append(expression.clone());
            lookup_ident.insert(name, expr_handle);
        }
        // read parameter list
        let mut arguments = Vec::new();
        lexer.expect(Token::Paren('('))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren(')')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next(),
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let binding = self.parse_varying_binding(lexer)?;
            let (param_name, param_type, _access) =
                self.parse_variable_ident_decl(lexer, &mut module.types, &mut module.constants)?;
            let param_index = arguments.len() as u32;
            let expression_token =
                expressions.append(crate::Expression::FunctionArgument(param_index));
            lookup_ident.insert(param_name, expression_token);
            arguments.push(crate::FunctionArgument {
                name: Some(param_name.to_string()),
                ty: param_type,
                binding,
            });
            ready = lexer.skip(Token::Separator(','));
        }
        // read return type
        let result = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let binding = self.parse_varying_binding(lexer)?;
            let (ty, _access) =
                self.parse_type_decl(lexer, None, &mut module.types, &mut module.constants)?;
            Some(crate::FunctionResult { ty, binding })
        } else {
            None
        };

        let mut fun = crate::Function {
            name: Some(fun_name.to_string()),
            arguments,
            result,
            local_variables: Arena::new(),
            expressions,
            named_expressions: crate::NamedExpressions::default(),
            body: Vec::new(),
        };

        // read body
        let mut typifier = super::Typifier::new();
        let mut named_expressions = crate::FastHashMap::default();
        fun.body = self.parse_block(
            lexer,
            StatementContext {
                lookup_ident: &mut lookup_ident,
                typifier: &mut typifier,
                variables: &mut fun.local_variables,
                expressions: &mut fun.expressions,
                named_expressions: &mut named_expressions,
                types: &mut module.types,
                constants: &mut module.constants,
                global_vars: &module.global_variables,
                functions: &module.functions,
                arguments: &fun.arguments,
            },
            true,
        )?;
        // fixup the IR
        ensure_block_returns(&mut fun.body);
        // done
        self.scopes.pop();

        // Set named expressions after block parsing ends
        fun.named_expressions = named_expressions;

        Ok((fun, fun_name))
    }

    fn parse_global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &mut FastHashMap<&'a str, crate::Expression>,
    ) -> Result<bool, Error<'a>> {
        // read attributes
        let mut binding = None;
        // Perspective is the default qualifier.
        let mut stage = None;
        let mut is_block = false;
        let mut workgroup_size = [0u32; 3];
        let mut early_depth_test = None;

        if lexer.skip(Token::DoubleParen('[')) {
            let (mut bind_index, mut bind_group) = (None, None);
            self.scopes.push(Scope::Attribute);
            loop {
                match lexer.next_ident_with_span()? {
                    ("binding", _) => {
                        lexer.expect(Token::Paren('('))?;
                        bind_index = Some(lexer.next_uint_literal()?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    ("block", _) => {
                        is_block = true;
                    }
                    ("group", _) => {
                        lexer.expect(Token::Paren('('))?;
                        bind_group = Some(lexer.next_uint_literal()?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    ("stage", _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (ident, ident_span) = lexer.next_ident_with_span()?;
                        stage = Some(conv::map_shader_stage(ident, ident_span)?);
                        lexer.expect(Token::Paren(')'))?;
                    }
                    ("workgroup_size", _) => {
                        lexer.expect(Token::Paren('('))?;
                        for (i, size) in workgroup_size.iter_mut().enumerate() {
                            *size = lexer.next_uint_literal()?;
                            match lexer.next() {
                                (Token::Paren(')'), _) => break,
                                (Token::Separator(','), _) if i != 2 => (),
                                other => {
                                    return Err(Error::Unexpected(
                                        other,
                                        ExpectedToken::WorkgroupSizeSeparator,
                                    ))
                                }
                            }
                        }
                        for size in workgroup_size.iter_mut() {
                            if *size == 0 {
                                *size = 1;
                            }
                        }
                    }
                    ("early_depth_test", _) => {
                        let conservative = if lexer.skip(Token::Paren('(')) {
                            let (ident, ident_span) = lexer.next_ident_with_span()?;
                            let value = conv::map_conservative_depth(ident, ident_span)?;
                            lexer.expect(Token::Paren(')'))?;
                            Some(value)
                        } else {
                            None
                        };
                        early_depth_test = Some(crate::EarlyDepthTest { conservative });
                    }
                    (_, word_span) => return Err(Error::UnknownAttribute(word_span)),
                }
                match lexer.next() {
                    (Token::DoubleParen(']'), _) => {
                        break;
                    }
                    (Token::Separator(','), _) => {}
                    other => {
                        return Err(Error::Unexpected(other, ExpectedToken::AttributeSeparator))
                    }
                }
            }
            if let (Some(group), Some(index)) = (bind_group, bind_index) {
                binding = Some(crate::ResourceBinding {
                    group,
                    binding: index,
                });
            }
            self.scopes.pop();
        }

        // read items
        match lexer.next() {
            (Token::Separator(';'), _) => {}
            (Token::Word("struct"), _) => {
                let name = lexer.next_ident()?;
                let (members, span) =
                    self.parse_struct_body(lexer, &mut module.types, &mut module.constants)?;
                let ty = module.types.fetch_or_append(crate::Type {
                    name: Some(name.to_string()),
                    inner: crate::TypeInner::Struct {
                        top_level: is_block,
                        members,
                        span,
                    },
                });
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            (Token::Word("type"), _) => {
                let name = lexer.next_ident()?;
                lexer.expect(Token::Operation('='))?;
                let (ty, _access) = self.parse_type_decl(
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            (Token::Word("let"), _) => {
                let (name, explicit_ty, _access) = self.parse_variable_ident_decl(
                    lexer,
                    &mut module.types,
                    &mut module.constants,
                )?;
                lexer.expect(Token::Operation('='))?;
                let first_token_span = lexer.next();
                let const_handle = self.parse_const_expression_impl(
                    first_token_span,
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                let con = &module.constants[const_handle];
                let type_match = match con.inner {
                    crate::ConstantInner::Scalar { width, value } => {
                        module.types[explicit_ty].inner
                            == crate::TypeInner::Scalar {
                                kind: value.scalar_kind(),
                                width,
                            }
                    }
                    crate::ConstantInner::Composite { ty, components: _ } => ty == explicit_ty,
                };
                if !type_match {
                    return Err(Error::LetTypeMismatch(name, explicit_ty));
                }
                //TODO: check `ty` against `const_handle`.
                lexer.expect(Token::Separator(';'))?;
                lookup_global_expression.insert(name, crate::Expression::Constant(const_handle));
            }
            (Token::Word("var"), _) => {
                let pvar =
                    self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                let class = match pvar.class {
                    Some(c) => c,
                    None => match module.types[pvar.ty].inner {
                        crate::TypeInner::Struct { .. } if binding.is_some() => {
                            if pvar.access.is_empty() {
                                crate::StorageClass::Uniform
                            } else {
                                crate::StorageClass::Storage
                            }
                        }
                        crate::TypeInner::Array { .. } if binding.is_some() => {
                            crate::StorageClass::Storage
                        }
                        crate::TypeInner::Image { .. } | crate::TypeInner::Sampler { .. } => {
                            crate::StorageClass::Handle
                        }
                        _ => crate::StorageClass::Private,
                    },
                };
                let var_handle = module.global_variables.append(crate::GlobalVariable {
                    name: Some(pvar.name.to_owned()),
                    class,
                    binding: binding.take(),
                    ty: pvar.ty,
                    init: pvar.init,
                    storage_access: pvar.access,
                });
                lookup_global_expression
                    .insert(pvar.name, crate::Expression::GlobalVariable(var_handle));
            }
            (Token::Word("fn"), _) => {
                let (function, name) =
                    self.parse_function_decl(lexer, module, lookup_global_expression)?;
                match stage {
                    Some(stage) => module.entry_points.push(crate::EntryPoint {
                        name: name.to_string(),
                        stage,
                        early_depth_test,
                        workgroup_size,
                        function,
                    }),
                    None => {
                        module.functions.append(function);
                    }
                }
            }
            (Token::End, _) => return Ok(false),
            other => return Err(Error::Unexpected(other, ExpectedToken::GlobalItem)),
        }

        match binding {
            None => Ok(true),
            // we had the attribute but no var?
            Some(_) => Err(Error::Other),
        }
    }

    pub fn parse(&mut self, source: &str) -> Result<crate::Module, ParseError> {
        self.scopes.clear();
        self.lookup_type.clear();
        self.layouter.clear();

        let mut module = crate::Module::default();
        let mut lexer = Lexer::new(source);
        let mut lookup_global_expression = FastHashMap::default();
        loop {
            match self.parse_global_decl(&mut lexer, &mut module, &mut lookup_global_expression) {
                Err(error) => return Err(error.as_parse_error(lexer.source)),
                Ok(true) => {}
                Ok(false) => {
                    if !self.scopes.is_empty() {
                        log::error!("Reached the end of file, but scopes are not closed");
                        return Err(Error::Other.as_parse_error(lexer.source));
                    };
                    module.apply_common_default_interpolation();
                    return Ok(module);
                }
            }
        }
    }
}

pub fn parse_str(source: &str) -> Result<crate::Module, ParseError> {
    Parser::new().parse(source)
}

pub struct StringErrorBuffer {
    buf: Vec<u8>,
}

impl StringErrorBuffer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn into_string(self) -> String {
        String::from_utf8(self.buf).unwrap()
    }
}

impl Write for StringErrorBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.extend(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl WriteColor for StringErrorBuffer {
    fn supports_color(&self) -> bool {
        false
    }

    fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}
