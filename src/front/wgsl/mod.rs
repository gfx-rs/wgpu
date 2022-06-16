/*!
Frontend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod construction;
mod conv;
mod lexer;
mod number;
#[cfg(test)]
mod tests;

use crate::{
    arena::{Arena, Handle, UniqueArena},
    proc::{
        ensure_block_returns, Alignment, Layouter, ResolveContext, ResolveError, TypeResolution,
    },
    span::SourceLocation,
    span::Span as NagaSpan,
    ConstantInner, FastHashMap, ScalarValue,
};

use self::{lexer::Lexer, number::Number};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, ColorSpec, StandardStream, WriteColor},
    },
};
use std::{
    borrow::Cow,
    convert::TryFrom,
    io::{self, Write},
    ops,
};
use thiserror::Error;

type Span = ops::Range<usize>;
type TokenSpan<'a> = (Token<'a>, Span);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    Paren(char),
    Attribute,
    Number(Result<Number, NumberError>),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    AssignmentOperation(char),
    IncrementOperation,
    DecrementOperation,
    Arrow,
    Unknown(char),
    Trivia,
    End,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NumberType {
    I32,
    U32,
    F32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    Number(NumberType),
    Integer,
    Constant,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: '}', identifier
    FieldName,
    /// Expected: attribute for a type
    TypeAttribute,
    /// Expected: ';', '{', word
    Statement,
    /// Expected: 'case', 'default', '}'
    SwitchItem,
    /// Expected: ',', ')'
    WorkgroupSizeSeparator,
    /// Expected: 'struct', 'let', 'var', 'type', ';', 'fn', eof
    GlobalItem,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NumberError {
    #[error("invalid numeric literal format")]
    Invalid,
    #[error("numeric literal not representable by target type")]
    NotRepresentable,
    #[error("unimplemented f16 type")]
    UnimplementedF16,
}

#[derive(Clone, Debug)]
pub enum Error<'a> {
    Unexpected(TokenSpan<'a>, ExpectedToken<'a>),
    UnexpectedComponents(Span),
    BadNumber(Span, NumberError),
    /// A negative signed integer literal where both signed and unsigned,
    /// but only non-negative literals are allowed.
    NegativeInt(Span),
    BadU32Constant(Span),
    BadMatrixScalarKind(Span, crate::ScalarKind, u8),
    BadAccessor(Span),
    BadTexture(Span),
    BadTypeCast {
        span: Span,
        from_type: String,
        to_type: String,
    },
    BadTextureSampleType {
        span: Span,
        kind: crate::ScalarKind,
        width: u8,
    },
    BadIncrDecrReferenceType(Span),
    InvalidResolve(ResolveError),
    InvalidForInitializer(Span),
    /// A break if appeared outside of a continuing block
    InvalidBreakIf(Span),
    InvalidGatherComponent(Span, u32),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownShaderStage(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    SizeAttributeTooLow(Span, u32),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    UnknownLocalFunction(Span),
    TypeNotConstructible(Span),
    TypeNotInferrable(Span),
    InitializationTypeMismatch(Span, String),
    MissingType(Span),
    MissingAttribute(&'static str, Span),
    InvalidAtomicPointer(Span),
    InvalidAtomicOperandType(Span),
    Pointer(&'static str, Span),
    NotPointer(Span),
    NotReference(&'static str, Span),
    ReservedKeyword(Span),
    Redefinition {
        previous: Span,
        current: Span,
    },
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
                                Token::Paren(c) => format!("'{}'", c),
                                Token::Attribute => "@".to_string(),
                                Token::Number(_) => "number".to_string(),
                                Token::Word(s) => s.to_string(),
                                Token::Operation(c) => format!("operation ('{}')", c),
                                Token::LogicalOperation(c) => format!("logical operation ('{}')", c),
                                Token::ShiftOperation(c) => format!("bitshift ('{}{}')", c, c),
                                Token::AssignmentOperation(c) if c=='<' || c=='>' => format!("bitshift ('{}{}=')", c, c),
                                Token::AssignmentOperation(c) => format!("operation ('{}=')", c),
                                Token::IncrementOperation => "increment operation".to_string(),
                                Token::DecrementOperation => "decrement operation".to_string(),
                                Token::Arrow => "->".to_string(),
                                Token::Unknown(c) => format!("unknown ('{}')", c),
                                Token::Trivia => "trivia".to_string(),
                                Token::End => "end".to_string(),
                            }
                        }
                        ExpectedToken::Identifier => "identifier".to_string(),
                        ExpectedToken::Number(ty) => {
                            match ty {
                                NumberType::I32 => "32-bit signed integer literal",
                                NumberType::U32 => "32-bit unsigned integer literal",
                                NumberType::F32 => "32-bit floating-point literal",
                            }.to_string()
                        },
                        ExpectedToken::Integer => "unsigned/signed integer literal".to_string(),
                        ExpectedToken::Constant => "constant".to_string(),
                        ExpectedToken::PrimaryExpression => "expression".to_string(),
                        ExpectedToken::FieldName => "field name or a closing curly bracket to signify the end of the struct".to_string(),
                        ExpectedToken::TypeAttribute => "type attribute".to_string(),
                        ExpectedToken::Statement => "statement".to_string(),
                        ExpectedToken::SwitchItem => "switch item ('case' or 'default') or a closing curly bracket to signify the end of the switch statement ('}')".to_string(),
                        ExpectedToken::WorkgroupSizeSeparator => "workgroup size separator (',') or a closing parenthesis".to_string(),
                        ExpectedToken::GlobalItem => "global item ('struct', 'let', 'var', 'type', ';', 'fn') or the end of the file".to_string(),
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
            Error::UnexpectedComponents(ref bad_span) => ParseError {
                message: "unexpected components".to_string(),
                labels: vec![(bad_span.clone(), "unexpected components".into())],
                notes: vec![],
            },
            Error::BadNumber(ref bad_span, ref err) => ParseError {
                message: format!(
                    "{}: `{}`",
                    err,&source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), err.to_string().into())],
                notes: vec![],
            },
            Error::NegativeInt(ref bad_span) => ParseError {
                message: format!(
                    "expected non-negative integer literal, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected non-negative integer".into())],
                notes: vec![],
            },
            Error::BadU32Constant(ref bad_span) => ParseError {
                message: format!(
                    "expected unsigned integer constant expression, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected unsigned integer".into())],
                notes: vec![],
            },
            Error::BadMatrixScalarKind(
                ref span,
                kind,
                width,
             ) => ParseError {
                message: format!("matrix scalar type must be floating-point, but found `{}`", kind.to_wgsl(width)),
                labels: vec![(span.clone(), "must be floating-point (e.g. `f32`)".into())],
                notes: vec![],
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
            Error::BadIncrDecrReferenceType(ref span) => ParseError {
                message: "increment/decrement operation requires reference type to be one of i32 or u32".to_string(),
                labels: vec![(span.clone(), "must be a reference type of i32 or u32".into())],
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
            Error::InvalidResolve(ref resolve_error) => ParseError {
                message: resolve_error.to_string(),
                labels: vec![],
                notes: vec![],
            },
            Error::InvalidForInitializer(ref bad_span) => ParseError {
                message: format!("for(;;) initializer is not an assignment or a function call: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "not an assignment or function call".into())],
                notes: vec![],
            },
            Error::InvalidBreakIf(ref bad_span) => ParseError {
                message: "A break if is only allowed in a continuing block".to_string(),
                labels: vec![(bad_span.clone(), "not in a continuing block".into())],
                notes: vec![],
            },
            Error::InvalidGatherComponent(ref bad_span, component) => ParseError {
                message: format!("textureGather component {} doesn't exist, must be 0, 1, 2, or 3", component),
                labels: vec![(bad_span.clone(), "invalid component".into())],
                notes: vec![],
            },
            Error::InvalidConstructorComponentType(ref bad_span, component) => ParseError {
                message: format!("invalid type for constructor component at index [{}]", component),
                labels: vec![(bad_span.clone(), "invalid component type".into())],
                notes: vec![],
            },
            Error::InvalidIdentifierUnderscore(ref bad_span) => ParseError {
                message: "Identifier can't be '_'".to_string(),
                labels: vec![(bad_span.clone(), "invalid identifier".into())],
                notes: vec!["Use phony assignment instead ('_ =' notice the absence of 'let' or 'var')".to_string()],
            },
            Error::ReservedIdentifierPrefix(ref bad_span) => ParseError {
                message: format!("Identifier starts with a reserved prefix: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "invalid identifier".into())],
                notes: vec![],
            },
            Error::UnknownAddressSpace(ref bad_span) => ParseError {
                message: format!("unknown address space: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown address space".into())],
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
            Error::UnknownAccess(ref bad_span) => ParseError {
                message: format!("unknown access: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown access".into())],
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
            Error::SizeAttributeTooLow(ref bad_span, min_size) => ParseError {
                message: format!("struct member size must be at least {}", min_size),
                labels: vec![(bad_span.clone(), format!("must be at least {}", min_size).into())],
                notes: vec![],
            },
            Error::AlignAttributeTooLow(ref bad_span, min_align) => ParseError {
                message: format!("struct member alignment must be at least {}", min_align),
                labels: vec![(bad_span.clone(), format!("must be at least {}", min_align).into())],
                notes: vec![],
            },
            Error::NonPowerOfTwoAlignAttribute(ref bad_span) => ParseError {
                message: "struct member alignment must be a power of 2".to_string(),
                labels: vec![(bad_span.clone(), "must be a power of 2".into())],
                notes: vec![],
            },
            Error::InconsistentBinding(ref span) => ParseError {
                message: "input/output binding is not consistent".to_string(),
                labels: vec![(span.clone(), "input/output binding is not consistent".into())],
                notes: vec![],
            },
            Error::UnknownLocalFunction(ref span) => ParseError {
                message: format!("unknown local function `{}`", &source[span.clone()]),
                labels: vec![(span.clone(), "unknown local function".into())],
                notes: vec![],
            },
            Error::TypeNotConstructible(ref span) => ParseError {
                message: format!("type `{}` is not constructible", &source[span.clone()]),
                labels: vec![(span.clone(), "type is not constructible".into())],
                notes: vec![],
            },
            Error::TypeNotInferrable(ref span) => ParseError {
                message: "type can't be inferred".to_string(),
                labels: vec![(span.clone(), "type can't be inferred".into())],
                notes: vec![],
            },
            Error::InitializationTypeMismatch(ref name_span, ref expected_ty) => ParseError {
                message: format!("the type of `{}` is expected to be `{}`", &source[name_span.clone()], expected_ty),
                labels: vec![(name_span.clone(), format!("definition of `{}`", &source[name_span.clone()]).into())],
                notes: vec![],
            },
            Error::MissingType(ref name_span) => ParseError {
                message: format!("variable `{}` needs a type", &source[name_span.clone()]),
                labels: vec![(name_span.clone(), format!("definition of `{}`", &source[name_span.clone()]).into())],
                notes: vec![],
            },
            Error::MissingAttribute(name, ref name_span) => ParseError {
                message: format!("variable `{}` needs a '{}' attribute", &source[name_span.clone()], name),
                labels: vec![(name_span.clone(), format!("definition of `{}`", &source[name_span.clone()]).into())],
                notes: vec![],
            },
            Error::InvalidAtomicPointer(ref span) => ParseError {
                message: "atomic operation is done on a pointer to a non-atomic".to_string(),
                labels: vec![(span.clone(), "atomic pointer is invalid".into())],
                notes: vec![],
            },
            Error::InvalidAtomicOperandType(ref span) => ParseError {
                message: "atomic operand type is inconsistent with the operation".to_string(),
                labels: vec![(span.clone(), "atomic operand type is invalid".into())],
                notes: vec![],
            },
            Error::NotPointer(ref span) => ParseError {
                message: "the operand of the `*` operator must be a pointer".to_string(),
                labels: vec![(span.clone(), "expression is not a pointer".into())],
                notes: vec![],
            },
            Error::NotReference(what, ref span) => ParseError {
                message: format!("{} must be a reference", what),
                labels: vec![(span.clone(), "expression is not a reference".into())],
                notes: vec![],
            },
            Error::Pointer(what, ref span) => ParseError {
                message: format!("{} must not be a pointer", what),
                labels: vec![(span.clone(), "expression is a pointer".into())],
                notes: vec![],
            },
            Error::ReservedKeyword(ref name_span) => ParseError {
                message: format!("name `{}` is a reserved keyword", &source[name_span.clone()]),
                labels: vec![(name_span.clone(), format!("definition of `{}`", &source[name_span.clone()]).into())],
                notes: vec![],
            },
            Error::Redefinition { ref previous, ref current } => ParseError {
                message: format!("redefinition of `{}`", &source[current.clone()]),
                labels: vec![(current.clone(), format!("redefinition of `{}`", &source[current.clone()]).into()),
                             (previous.clone(), format!("previous definition of `{}`", &source[previous.clone()]).into())
                ],
                notes: vec![],
            },
            Error::Other => ParseError {
                message: "other error".to_string(),
                labels: vec![],
                notes: vec![],
            },
        }
    }
}

impl crate::StorageFormat {
    const fn to_wgsl(self) -> &'static str {
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
    fn to_wgsl(
        &self,
        types: &UniqueArena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> String {
        use crate::TypeInner as Ti;

        match *self {
            Ti::Scalar { kind, width } => kind.to_wgsl(width),
            Ti::Vector { size, kind, width } => {
                format!("vec{}<{}>", size as u32, kind.to_wgsl(width))
            }
            Ti::Matrix {
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
            Ti::Atomic { kind, width } => {
                format!("atomic<{}>", kind.to_wgsl(width))
            }
            Ti::Pointer { base, .. } => {
                let base = &types[base];
                let name = base.name.as_deref().unwrap_or("unknown");
                format!("ptr<{}>", name)
            }
            Ti::ValuePointer { kind, width, .. } => {
                format!("ptr<{}>", kind.to_wgsl(width))
            }
            Ti::Array { base, size, .. } => {
                let member_type = &types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => {
                        let size = constants[size].name.as_deref().unwrap_or("unknown");
                        format!("array<{}, {}>", base, size)
                    }
                    crate::ArraySize::Dynamic => format!("array<{}>", base),
                }
            }
            Ti::Struct { .. } => {
                // TODO: Actually output the struct?
                "struct".to_string()
            }
            Ti::Image {
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
                    crate::ImageClass::Depth { multi: false } => "_depth",
                    crate::ImageClass::Depth { multi: true } => "_depth_multisampled",
                    crate::ImageClass::Sampled { multi: false, .. }
                    | crate::ImageClass::Storage { .. } => "",
                };

                let type_in_brackets = match class {
                    crate::ImageClass::Sampled { kind, .. } => {
                        // Note: The only valid widths are 4 bytes wide.
                        // The lexer has already verified this, so we can safely assume it here.
                        // https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                        let element_type = kind.to_wgsl(4);
                        format!("<{}>", element_type)
                    }
                    crate::ImageClass::Depth { multi: _ } => String::new(),
                    crate::ImageClass::Storage { format, access } => {
                        if access.contains(crate::StorageAccess::STORE) {
                            format!("<{},write>", format.to_wgsl())
                        } else {
                            format!("<{}>", format.to_wgsl())
                        }
                    }
                };

                format!(
                    "texture{}{}{}{}",
                    class_suffix, dim_suffix, array_suffix, type_in_brackets
                )
            }
            Ti::Sampler { .. } => "sampler".to_string(),
            Ti::BindingArray { base, size, .. } => {
                let member_type = &types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => {
                        let size = constants[size].name.as_deref().unwrap_or("unknown");
                        format!("binding_array<{}, {}>", base, size)
                    }
                    crate::ArraySize::Dynamic => format!("binding_array<{}>", base),
                }
            }
        }
    }
}

mod type_inner_tests {
    #[test]
    fn to_wgsl() {
        let mut types = crate::UniqueArena::new();
        let mut constants = crate::Arena::new();
        let c = constants.append(
            crate::Constant {
                name: Some("C".to_string()),
                specialization: None,
                inner: crate::ConstantInner::Scalar {
                    width: 4,
                    value: crate::ScalarValue::Uint(32),
                },
            },
            Default::default(),
        );

        let mytype1 = types.insert(
            crate::Type {
                name: Some("MyType1".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );
        let mytype2 = types.insert(
            crate::Type {
                name: Some("MyType2".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );

        let array = crate::TypeInner::Array {
            base: mytype1,
            stride: 4,
            size: crate::ArraySize::Constant(c),
        };
        assert_eq!(array.to_wgsl(&types, &constants), "array<MyType1, C>");

        let mat = crate::TypeInner::Matrix {
            rows: crate::VectorSize::Quad,
            columns: crate::VectorSize::Bi,
            width: 8,
        };
        assert_eq!(mat.to_wgsl(&types, &constants), "mat2x4<f64>");

        let ptr = crate::TypeInner::Pointer {
            base: mytype2,
            space: crate::AddressSpace::Storage {
                access: crate::StorageAccess::default(),
            },
        };
        assert_eq!(ptr.to_wgsl(&types, &constants), "ptr<MyType2>");

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
            class: crate::ImageClass::Depth { multi: false },
        };
        assert_eq!(img2.to_wgsl(&types, &constants), "texture_depth_cube_array");

        let img3 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Depth { multi: true },
        };
        assert_eq!(
            img3.to_wgsl(&types, &constants),
            "texture_depth_multisampled_2d"
        );

        let array = crate::TypeInner::BindingArray {
            base: mytype1,
            size: crate::ArraySize::Constant(c),
        };
        assert_eq!(
            array.to_wgsl(&types, &constants),
            "binding_array<MyType1, C>"
        );
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
impl<'a> StringValueLookup<'a> for FastHashMap<&'a str, TypedExpression> {
    type Value = TypedExpression;
    fn lookup(&self, key: &'a str, span: Span) -> Result<Self::Value, Error<'a>> {
        self.get(key).cloned().ok_or(Error::UnknownIdent(span, key))
    }
}

struct StatementContext<'input, 'temp, 'out> {
    lookup_ident: &'temp mut FastHashMap<&'input str, TypedExpression>,
    typifier: &'temp mut super::Typifier,
    variables: &'out mut Arena<crate::LocalVariable>,
    expressions: &'out mut Arena<crate::Expression>,
    named_expressions: &'out mut FastHashMap<Handle<crate::Expression>, String>,
    types: &'out mut UniqueArena<crate::Type>,
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
    lookup_ident: &'temp FastHashMap<&'input str, TypedExpression>,
    typifier: &'temp mut super::Typifier,
    expressions: &'out mut Arena<crate::Expression>,
    types: &'out mut UniqueArena<crate::Type>,
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
        image: Handle<crate::Expression>,
        span: Span,
    ) -> Result<SamplingContext, Error<'a>> {
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
        ) -> Result<TypedExpression, Error<'a>>,
    ) -> Result<TypedExpression, Error<'a>> {
        let start = lexer.current_byte_offset() as u32;
        let mut accumulator = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            // Binary expressions always apply the load rule to their operands.
            let mut left = self.apply_load_rule(accumulator);
            let unloaded_right = parser(lexer, self.reborrow())?;
            let right = self.apply_load_rule(unloaded_right);
            let end = lexer.current_byte_offset() as u32;
            left = self.expressions.append(
                crate::Expression::Binary { op, left, right },
                NagaSpan::new(start, end),
            );
            // Binary expressions never produce references.
            accumulator = TypedExpression::non_reference(left);
        }
        Ok(accumulator)
    }

    fn parse_binary_splat_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            ExpressionContext<'a, '_, '_>,
        ) -> Result<TypedExpression, Error<'a>>,
    ) -> Result<TypedExpression, Error<'a>> {
        let start = lexer.current_byte_offset() as u32;
        let mut accumulator = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            // Binary expressions always apply the load rule to their operands.
            let mut left = self.apply_load_rule(accumulator);
            let unloaded_right = parser(lexer, self.reborrow())?;
            let mut right = self.apply_load_rule(unloaded_right);
            let end = lexer.current_byte_offset() as u32;

            // Insert splats, if needed by the non-'*' operations.
            // (`BinaryOperator::Multiply` handles splats itself.)
            if op != crate::BinaryOperator::Multiply {
                let left_size = match *self.resolve_type(left)? {
                    crate::TypeInner::Vector { size, .. } => Some(size),
                    _ => None,
                };
                match (left_size, self.resolve_type(right)?) {
                    (Some(size), &crate::TypeInner::Scalar { .. }) => {
                        right = self.expressions.append(
                            crate::Expression::Splat { size, value: right },
                            self.expressions.get_span(right),
                        );
                    }
                    (None, &crate::TypeInner::Vector { size, .. }) => {
                        left = self.expressions.append(
                            crate::Expression::Splat { size, value: left },
                            self.expressions.get_span(left),
                        );
                    }
                    _ => {}
                }
            }
            accumulator = TypedExpression::non_reference(self.expressions.append(
                crate::Expression::Binary { op, left, right },
                NagaSpan::new(start, end),
            ));
        }
        Ok(accumulator)
    }

    /// Add a single expression to the expression table that is not covered by `self.emitter`.
    ///
    /// This is useful for `CallResult` and `AtomicResult` expressions, which should not be covered by
    /// `Emit` statements.
    fn interrupt_emitter(
        &mut self,
        expression: crate::Expression,
        span: NagaSpan,
    ) -> Handle<crate::Expression> {
        self.block.extend(self.emitter.finish(self.expressions));
        let result = self.expressions.append(expression, span);
        self.emitter.start(self.expressions);
        result
    }

    /// Apply the WGSL Load Rule to `expr`.
    ///
    /// If `expr` is has type `ref<SC, T, A>`, perform a load to produce a value of type
    /// `T`. Otherwise, return `expr` unchanged.
    fn apply_load_rule(&mut self, expr: TypedExpression) -> Handle<crate::Expression> {
        if expr.is_reference {
            let load = crate::Expression::Load {
                pointer: expr.handle,
            };
            let span = self.expressions.get_span(expr.handle);
            self.expressions.append(load, span)
        } else {
            expr.handle
        }
    }

    /// Creates a zero value constant of type `ty`
    ///
    /// Returns `None` if the given `ty` is not a constructible type
    fn create_zero_value_constant(
        &mut self,
        ty: Handle<crate::Type>,
    ) -> Option<Handle<crate::Constant>> {
        let inner = match self.types[ty].inner {
            crate::TypeInner::Scalar { kind, width } => {
                let value = match kind {
                    crate::ScalarKind::Sint => crate::ScalarValue::Sint(0),
                    crate::ScalarKind::Uint => crate::ScalarValue::Uint(0),
                    crate::ScalarKind::Float => crate::ScalarValue::Float(0.),
                    crate::ScalarKind::Bool => crate::ScalarValue::Bool(false),
                };
                crate::ConstantInner::Scalar { width, value }
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar { width, kind },
                    },
                    Default::default(),
                );
                let component = self.create_zero_value_constant(scalar_ty);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..size as u8).map(|_| component).collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let vec_ty = self.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Vector {
                            width,
                            kind: crate::ScalarKind::Float,
                            size: rows,
                        },
                    },
                    Default::default(),
                );
                let component = self.create_zero_value_constant(vec_ty);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..columns as u8)
                        .map(|_| component)
                        .collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(size),
                ..
            } => {
                let component = self.create_zero_value_constant(base);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..self.constants[size].to_array_length().unwrap())
                        .map(|_| component)
                        .collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Struct { ref members, .. } => {
                let members = members.clone();
                crate::ConstantInner::Composite {
                    ty,
                    components: members
                        .iter()
                        .map(|member| self.create_zero_value_constant(member.ty))
                        .collect::<Option<_>>()?,
                }
            }
            _ => return None,
        };

        let constant = self.constants.fetch_or_append(
            crate::Constant {
                name: None,
                specialization: None,
                inner,
            },
            crate::Span::default(),
        );
        Some(constant)
    }
}

/// A Naga [`Expression`] handle, with WGSL type information.
///
/// Naga and WGSL types are very close, but Naga lacks WGSL's 'reference' types,
/// which we need to know to apply the Load Rule. This struct carries a Naga
/// `Handle<Expression>` along with enough information to determine its WGSL type.
///
/// [`Expression`]: crate::Expression
#[derive(Debug, Copy, Clone)]
struct TypedExpression {
    /// The handle of the Naga expression.
    handle: Handle<crate::Expression>,

    /// True if this expression's WGSL type is a reference.
    ///
    /// When this is true, `handle` must be a pointer.
    is_reference: bool,
}

impl TypedExpression {
    const fn non_reference(handle: Handle<crate::Expression>) -> TypedExpression {
        TypedExpression {
            handle,
            is_reference: false,
        }
    }
}

enum Composition {
    Single(u32),
    Multi(crate::VectorSize, [crate::SwizzleComponent; 4]),
}

impl Composition {
    const fn letter_component(letter: char) -> Option<crate::SwizzleComponent> {
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
    // Although WGSL nas no type attributes at the moment, it had them in the past
    // (`[[stride]]`) and may as well acquire some again in the future.
    // Therefore, we are leaving the plumbing in for now.
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
    UnaryExpr,
    GeneralExpr,
}

type LocalFunctionCall = (Handle<crate::Function>, Vec<Handle<crate::Expression>>);

#[derive(Default)]
struct BindingParser {
    location: Option<u32>,
    built_in: Option<crate::BuiltIn>,
    interpolation: Option<crate::Interpolation>,
    sampling: Option<crate::Sampling>,
    invariant: bool,
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
                self.location = Some(Parser::parse_non_negative_i32_literal(lexer)?);
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
            "invariant" => self.invariant = true,
            _ => return Err(Error::UnknownAttribute(name_span)),
        }
        Ok(())
    }

    const fn finish<'a>(self, span: Span) -> Result<Option<crate::Binding>, Error<'a>> {
        match (
            self.location,
            self.built_in,
            self.interpolation,
            self.sampling,
            self.invariant,
        ) {
            (None, None, None, None, false) => Ok(None),
            (Some(location), None, interpolation, sampling, false) => {
                // Before handing over the completed `Module`, we call
                // `apply_default_interpolation` to ensure that the interpolation and
                // sampling have been explicitly specified on all vertex shader output and fragment
                // shader input user bindings, so leaving them potentially `None` here is fine.
                Ok(Some(crate::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                }))
            }
            (None, Some(crate::BuiltIn::Position { .. }), None, None, invariant) => {
                Ok(Some(crate::Binding::BuiltIn(crate::BuiltIn::Position {
                    invariant,
                })))
            }
            (None, Some(built_in), None, None, false) => {
                Ok(Some(crate::Binding::BuiltIn(built_in)))
            }
            (_, _, _, _, _) => Err(Error::InconsistentBinding(span)),
        }
    }
}

struct ParsedVariable<'a> {
    name: &'a str,
    name_span: Span,
    space: Option<crate::AddressSpace>,
    ty: Handle<crate::Type>,
    init: Option<Handle<crate::Constant>>,
}

struct CalledFunction {
    result: Option<Handle<crate::Expression>>,
}

#[derive(Clone, Debug)]
pub struct ParseError {
    message: String,
    labels: Vec<(Span, Cow<'static, str>)>,
    notes: Vec<String>,
}

impl ParseError {
    pub fn labels(&self) -> impl Iterator<Item = (Span, &str)> + ExactSizeIterator + '_ {
        self.labels
            .iter()
            .map(|&(ref span, ref msg)| (span.clone(), msg.as_ref()))
    }

    pub fn message(&self) -> &str {
        &self.message
    }

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
        self.emit_to_stderr_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to standard error stream.
    pub fn emit_to_stderr_with_path(&self, source: &str, path: &str) {
        let files = SimpleFile::new(path, source);
        let config = codespan_reporting::term::Config::default();
        let writer = StandardStream::stderr(ColorChoice::Auto);
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

    /// Returns a [`SourceLocation`] for the first label in the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        self.labels
            .get(0)
            .map(|label| NagaSpan::new(label.0.start as u32, label.0.end as u32).location(source))
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
    scopes: Vec<(Scope, usize)>,
    module_scope_identifiers: FastHashMap<String, Span>,
    lookup_type: FastHashMap<String, Handle<crate::Type>>,
    layouter: Layouter,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            scopes: Vec::new(),
            module_scope_identifiers: FastHashMap::default(),
            lookup_type: FastHashMap::default(),
            layouter: Default::default(),
        }
    }

    fn reset(&mut self) {
        self.scopes.clear();
        self.module_scope_identifiers.clear();
        self.lookup_type.clear();
        self.layouter.clear();
    }

    fn push_scope(&mut self, scope: Scope, lexer: &Lexer<'_>) {
        self.scopes.push((scope, lexer.current_byte_offset()));
    }

    fn pop_scope(&mut self, lexer: &Lexer<'_>) -> Span {
        let (_, initial) = self.scopes.pop().unwrap();
        lexer.span_from(initial)
    }

    fn peek_scope(&mut self, lexer: &Lexer<'_>) -> Span {
        let &(_, initial) = self.scopes.last().unwrap();
        lexer.span_from(initial)
    }

    fn parse_switch_value<'a>(lexer: &mut Lexer<'a>, uint: bool) -> Result<i32, Error<'a>> {
        let token_span = lexer.next();
        match token_span.0 {
            Token::Number(Ok(Number::U32(num))) if uint => Ok(num as i32),
            Token::Number(Ok(Number::I32(num))) if !uint => Ok(num),
            Token::Number(Err(e)) => Err(Error::BadNumber(token_span.1, e)),
            _ => Err(Error::Unexpected(token_span, ExpectedToken::Integer)),
        }
    }

    /// Parse a non-negative signed integer literal.
    /// This is for attributes like `size`, `location` and others.
    fn parse_non_negative_i32_literal<'a>(lexer: &mut Lexer<'a>) -> Result<u32, Error<'a>> {
        match lexer.next() {
            (Token::Number(Ok(Number::I32(num))), span) => {
                u32::try_from(num).map_err(|_| Error::NegativeInt(span))
            }
            (Token::Number(Err(e)), span) => Err(Error::BadNumber(span, e)),
            other => Err(Error::Unexpected(
                other,
                ExpectedToken::Number(NumberType::I32),
            )),
        }
    }

    /// Parse a non-negative integer literal that may be either signed or unsigned.
    /// This is for the `workgroup_size` attribute and array lengths.
    /// Note: these values should be no larger than [`i32::MAX`], but this is not checked here.
    fn parse_generic_non_negative_int_literal<'a>(lexer: &mut Lexer<'a>) -> Result<u32, Error<'a>> {
        match lexer.next() {
            (Token::Number(Ok(Number::I32(num))), span) => {
                u32::try_from(num).map_err(|_| Error::NegativeInt(span))
            }
            (Token::Number(Ok(Number::U32(num))), _) => Ok(num),
            (Token::Number(Err(e)), span) => Err(Error::BadNumber(span, e)),
            other => Err(Error::Unexpected(
                other,
                ExpectedToken::Number(NumberType::I32),
            )),
        }
    }

    fn parse_atomic_pointer<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let (pointer, pointer_span) =
            lexer.capture_span(|lexer| self.parse_general_expression(lexer, ctx.reborrow()))?;
        // Check if the pointer expression is to an atomic.
        // The IR uses regular `Expression::Load` and `Statement::Store` for atomic load/stores,
        // and it will not catch the use of a non-atomic variable here.
        match *ctx.resolve_type(pointer)? {
            crate::TypeInner::Pointer { base, .. } => match ctx.types[base].inner {
                crate::TypeInner::Atomic { .. } => Ok(pointer),
                ref other => {
                    log::error!("Pointer type to {:?} passed to atomic op", other);
                    Err(Error::InvalidAtomicPointer(pointer_span))
                }
            },
            ref other => {
                log::error!("Type {:?} passed to atomic op", other);
                Err(Error::InvalidAtomicPointer(pointer_span))
            }
        }
    }

    /// Expects name to be peeked from lexer, does not consume if returns None.
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
        let _ = lexer.next();
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

    fn parse_atomic_helper<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        fun: crate::AtomicFunction,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        lexer.open_arguments()?;
        let pointer = self.parse_general_expression(lexer, ctx.reborrow())?;
        lexer.expect(Token::Separator(','))?;
        let ctx_span = ctx.reborrow();
        let (value, value_span) =
            lexer.capture_span(|lexer| self.parse_general_expression(lexer, ctx_span))?;
        lexer.close_arguments()?;

        let expression = match *ctx.resolve_type(value)? {
            crate::TypeInner::Scalar { kind, width } => crate::Expression::AtomicResult {
                kind,
                width,
                comparison: false,
            },
            _ => return Err(Error::InvalidAtomicOperandType(value_span)),
        };

        let span = NagaSpan::from(value_span);
        let result = ctx.interrupt_emitter(expression, span);
        ctx.block.push(
            crate::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            },
            span,
        );
        Ok(result)
    }

    /// Expects [`Scope::PrimaryExpr`] or [`Scope::SingularExpr`] on top; does not pop it.
    /// Expects `word` to be peeked (still in lexer), doesn't consume if returning None.
    fn parse_function_call_inner<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<CalledFunction>, Error<'a>> {
        assert!(self.scopes.last().is_some());
        let expr = if let Some(fun) = conv::map_relational_fun(name) {
            let _ = lexer.next();
            lexer.open_arguments()?;
            let argument = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::Relational { fun, argument }
        } else if let Some(axis) = conv::map_derivative_axis(name) {
            let _ = lexer.next();
            lexer.open_arguments()?;
            let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
            lexer.close_arguments()?;
            crate::Expression::Derivative { axis, expr }
        } else if let Some(fun) = conv::map_standard_fun(name) {
            let _ = lexer.next();
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
            let arg3 = if arg_count > 3 {
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
                arg3,
            }
        } else {
            match name {
                "bitcast" => {
                    let _ = lexer.next();
                    lexer.expect_generic_paren('<')?;
                    let (ty, type_span) = lexer.capture_span(|lexer| {
                        self.parse_type_decl(lexer, None, ctx.types, ctx.constants)
                    })?;
                    lexer.expect_generic_paren('>')?;

                    lexer.open_arguments()?;
                    let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;

                    let kind = match ctx.types[ty].inner {
                        crate::TypeInner::Scalar { kind, .. } => kind,
                        crate::TypeInner::Vector { kind, .. } => kind,
                        _ => {
                            return Err(Error::BadTypeCast {
                                from_type: format!("{:?}", ctx.resolve_type(expr)?),
                                span: type_span,
                                to_type: format!("{:?}", ctx.types[ty].inner),
                            })
                        }
                    };

                    crate::Expression::As {
                        expr,
                        kind,
                        convert: None,
                    }
                }
                "select" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let reject = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let accept = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let condition = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::Select {
                        condition,
                        accept,
                        reject,
                    }
                }
                "arrayLength" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let array = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::ArrayLength(array)
                }
                // atomics
                "atomicLoad" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let pointer = self.parse_atomic_pointer(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::Load { pointer }
                }
                "atomicAdd" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::Add,
                        ctx.reborrow(),
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicSub" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::Subtract,
                        ctx.reborrow(),
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicAnd" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::And,
                        ctx.reborrow(),
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicOr" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::InclusiveOr,
                        ctx.reborrow(),
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicXor" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::ExclusiveOr,
                        ctx.reborrow(),
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicMin" => {
                    let _ = lexer.next();
                    let handle =
                        self.parse_atomic_helper(lexer, crate::AtomicFunction::Min, ctx)?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicMax" => {
                    let _ = lexer.next();
                    let handle =
                        self.parse_atomic_helper(lexer, crate::AtomicFunction::Max, ctx)?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicExchange" => {
                    let _ = lexer.next();
                    let handle = self.parse_atomic_helper(
                        lexer,
                        crate::AtomicFunction::Exchange { compare: None },
                        ctx,
                    )?;
                    return Ok(Some(CalledFunction {
                        result: Some(handle),
                    }));
                }
                "atomicCompareExchangeWeak" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let pointer = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let cmp = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let (value, value_span) = lexer.capture_span(|lexer| {
                        self.parse_general_expression(lexer, ctx.reborrow())
                    })?;
                    lexer.close_arguments()?;

                    let expression = match *ctx.resolve_type(value)? {
                        crate::TypeInner::Scalar { kind, width } => {
                            crate::Expression::AtomicResult {
                                kind,
                                width,
                                comparison: true,
                            }
                        }
                        _ => return Err(Error::InvalidAtomicOperandType(value_span)),
                    };

                    let span = NagaSpan::from(self.peek_scope(lexer));
                    let result = ctx.interrupt_emitter(expression, span);
                    ctx.block.push(
                        crate::Statement::Atomic {
                            pointer,
                            fun: crate::AtomicFunction::Exchange { compare: Some(cmp) },
                            value,
                            result,
                        },
                        span,
                    );
                    return Ok(Some(CalledFunction {
                        result: Some(result),
                    }));
                }
                // texture sampling
                "textureSample" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Auto,
                        depth_ref: None,
                    }
                }
                "textureSampleLevel" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Exact(level),
                        depth_ref: None,
                    }
                }
                "textureSampleBias" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Bias(bias),
                        depth_ref: None,
                    }
                }
                "textureSampleGrad" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Gradient { x, y },
                        depth_ref: None,
                    }
                }
                "textureSampleCompare" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Auto,
                        depth_ref: Some(reference),
                    }
                }
                "textureSampleCompareLevel" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: None,
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Zero,
                        depth_ref: Some(reference),
                    }
                }
                "textureGather" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let component = if let (Token::Number(..), span) = lexer.peek() {
                        let index = Self::parse_non_negative_i32_literal(lexer)?;
                        lexer.expect(Token::Separator(','))?;
                        *crate::SwizzleComponent::XYZW
                            .get(index as usize)
                            .ok_or(Error::InvalidGatherComponent(span, index))?
                    } else {
                        crate::SwizzleComponent::X
                    };
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: Some(component),
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Zero,
                        depth_ref: None,
                    }
                }
                "textureGatherCompare" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let sampler_expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Separator(','))?;
                    let coordinate = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let sc = ctx.prepare_sampling(image, image_span)?;
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
                        sampler: sampler_expr,
                        gather: Some(crate::SwizzleComponent::X),
                        coordinate,
                        array_index,
                        offset,
                        level: crate::SampleLevel::Zero,
                        depth_ref: Some(reference),
                    }
                }
                "textureLoad" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let (image, image_span) =
                        self.parse_general_expression_with_span(lexer, ctx.reborrow())?;
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
                    let level = if class.is_mipmapped() {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    let sample = if class.is_multisampled() {
                        lexer.expect(Token::Separator(','))?;
                        Some(self.parse_general_expression(lexer, ctx.reborrow())?)
                    } else {
                        None
                    };
                    lexer.close_arguments()?;
                    crate::Expression::ImageLoad {
                        image,
                        coordinate,
                        array_index,
                        sample,
                        level,
                    }
                }
                "textureDimensions" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let image = self.parse_general_expression(lexer, ctx.reborrow())?;
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
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let image = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLevels,
                    }
                }
                "textureNumLayers" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let image = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumLayers,
                    }
                }
                "textureNumSamples" => {
                    let _ = lexer.next();
                    lexer.open_arguments()?;
                    let image = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;
                    crate::Expression::ImageQuery {
                        image,
                        query: crate::ImageQuery::NumSamples,
                    }
                }
                // other
                _ => {
                    let result =
                        match self.parse_local_function_call(lexer, name, ctx.reborrow())? {
                            Some((function, arguments)) => {
                                let span = NagaSpan::from(self.peek_scope(lexer));
                                ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                                let result = ctx.functions[function].result.as_ref().map(|_| {
                                    ctx.expressions
                                        .append(crate::Expression::CallResult(function), span)
                                });
                                ctx.emitter.start(ctx.expressions);
                                ctx.block.push(
                                    crate::Statement::Call {
                                        function,
                                        arguments,
                                        result,
                                    },
                                    span,
                                );
                                result
                            }
                            None => return Ok(None),
                        };
                    return Ok(Some(CalledFunction { result }));
                }
            }
        };
        let span = NagaSpan::from(self.peek_scope(lexer));
        let handle = ctx.expressions.append(expr, span);
        Ok(Some(CalledFunction {
            result: Some(handle),
        }))
    }

    fn parse_const_expression_impl<'a>(
        &mut self,
        first_token_span: TokenSpan<'a>,
        lexer: &mut Lexer<'a>,
        register_name: Option<&'a str>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.push_scope(Scope::ConstantExpr, lexer);
        let inner = match first_token_span {
            (Token::Word("true"), _) => crate::ConstantInner::boolean(true),
            (Token::Word("false"), _) => crate::ConstantInner::boolean(false),
            (Token::Number(num), _) => match num {
                Ok(Number::I32(num)) => crate::ConstantInner::Scalar {
                    value: crate::ScalarValue::Sint(num as i64),
                    width: 4,
                },
                Ok(Number::U32(num)) => crate::ConstantInner::Scalar {
                    value: crate::ScalarValue::Uint(num as u64),
                    width: 4,
                },
                Ok(Number::F32(num)) => crate::ConstantInner::Scalar {
                    value: crate::ScalarValue::Float(num as f64),
                    width: 4,
                },
                Ok(Number::AbstractInt(_) | Number::AbstractFloat(_)) => unreachable!(),
                Err(e) => return Err(Error::BadNumber(first_token_span.1, e)),
            },
            (Token::Word(name), name_span) => {
                // look for an existing constant first
                for (handle, var) in const_arena.iter() {
                    match var.name {
                        Some(ref string) if string == name => {
                            self.pop_scope(lexer);
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

        // Only set span if it's a named constant. Otherwise, the enclosing Expression should have
        // the span.
        let span = self.pop_scope(lexer);
        let handle = if let Some(name) = register_name {
            if crate::keywords::wgsl::RESERVED.contains(&name) {
                return Err(Error::ReservedKeyword(span));
            }
            const_arena.append(
                crate::Constant {
                    name: Some(name.to_string()),
                    specialization: None,
                    inner,
                },
                NagaSpan::from(span),
            )
        } else {
            const_arena.fetch_or_append(
                crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                },
                Default::default(),
            )
        };

        Ok(handle)
    }

    fn parse_const_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Constant>, Error<'a>> {
        self.parse_const_expression_impl(lexer.next(), lexer, None, type_arena, const_arena)
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<TypedExpression, Error<'a>> {
        // Will be popped inside match, possibly inside parse_function_call_inner or parse_construction
        self.push_scope(Scope::PrimaryExpr, lexer);
        let expr = match lexer.peek() {
            (Token::Paren('('), _) => {
                let _ = lexer.next();
                let (expr, _span) =
                    self.parse_general_expression_for_reference(lexer, ctx.reborrow())?;
                lexer.expect(Token::Paren(')'))?;
                self.pop_scope(lexer);
                expr
            }
            (Token::Word("true" | "false") | Token::Number(..), _) => {
                let const_handle = self.parse_const_expression(lexer, ctx.types, ctx.constants)?;
                let span = NagaSpan::from(self.pop_scope(lexer));
                TypedExpression::non_reference(
                    ctx.interrupt_emitter(crate::Expression::Constant(const_handle), span),
                )
            }
            (Token::Word(word), span) => {
                if let Some(definition) = ctx.lookup_ident.get(word) {
                    let _ = lexer.next();
                    self.pop_scope(lexer);

                    *definition
                } else if let Some(CalledFunction { result: Some(expr) }) =
                    self.parse_function_call_inner(lexer, word, ctx.reborrow())?
                {
                    //TODO: resolve the duplicate call in `parse_singular_expression`
                    self.pop_scope(lexer);
                    TypedExpression::non_reference(expr)
                } else {
                    let _ = lexer.next();
                    if let Some(expr) = construction::parse_construction(
                        self,
                        lexer,
                        word,
                        span.clone(),
                        ctx.reborrow(),
                    )? {
                        TypedExpression::non_reference(expr)
                    } else {
                        return Err(Error::UnknownIdent(span, word));
                    }
                }
            }
            other => return Err(Error::Unexpected(other, ExpectedToken::PrimaryExpression)),
        };
        Ok(expr)
    }

    fn parse_postfix<'a>(
        &mut self,
        span_start: usize,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
        expr: TypedExpression,
    ) -> Result<TypedExpression, Error<'a>> {
        // Parse postfix expressions, adjusting `handle` and `is_reference` along the way.
        //
        // Most postfix expressions don't affect `is_reference`: for example, `s.x` is a
        // reference whenever `s` is a reference. But swizzles (WGSL spec: "multiple
        // component selection") apply the load rule, converting references to values, so
        // those affect `is_reference` as well as `handle`.
        let TypedExpression {
            mut handle,
            mut is_reference,
        } = expr;
        let mut prefix_span = lexer.span_from(span_start);

        loop {
            // Step lightly around `resolve_type`'s mutable borrow.
            ctx.resolve_type(handle)?;

            // Find the type of the composite whose elements, components or members we're
            // accessing, skipping through references: except for swizzles, the `Access`
            // or `AccessIndex` expressions we'd generate are the same either way.
            //
            // Pointers, however, are not permitted. For error checks below, note whether
            // the base expression is a WGSL pointer.
            let temp_inner;
            let (composite, wgsl_pointer) = match *ctx.typifier.get(handle, ctx.types) {
                crate::TypeInner::Pointer { base, .. } => (&ctx.types[base].inner, !is_reference),
                crate::TypeInner::ValuePointer {
                    size: None,
                    kind,
                    width,
                    ..
                } => {
                    temp_inner = crate::TypeInner::Scalar { kind, width };
                    (&temp_inner, !is_reference)
                }
                crate::TypeInner::ValuePointer {
                    size: Some(size),
                    kind,
                    width,
                    ..
                } => {
                    temp_inner = crate::TypeInner::Vector { size, kind, width };
                    (&temp_inner, !is_reference)
                }
                ref other => (other, false),
            };

            let expression = match lexer.peek().0 {
                Token::Separator('.') => {
                    let _ = lexer.next();
                    let (name, name_span) = lexer.next_ident_with_span()?;

                    // WGSL doesn't allow accessing members on pointers, or swizzling
                    // them. But Naga IR doesn't distinguish pointers and references, so
                    // we must check here.
                    if wgsl_pointer {
                        return Err(Error::Pointer(
                            "the value accessed by a `.member` expression",
                            prefix_span,
                        ));
                    }

                    let access = match *composite {
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
                                Composition::Multi(size, pattern) => {
                                    // Once you apply the load rule, the expression is no
                                    // longer a reference.
                                    let current_expr = TypedExpression {
                                        handle,
                                        is_reference,
                                    };
                                    let vector = ctx.apply_load_rule(current_expr);
                                    is_reference = false;

                                    crate::Expression::Swizzle {
                                        size,
                                        vector,
                                        pattern,
                                    }
                                }
                                Composition::Single(index) => crate::Expression::AccessIndex {
                                    base: handle,
                                    index,
                                },
                            }
                        }
                        _ => return Err(Error::BadAccessor(name_span)),
                    };

                    access
                }
                Token::Paren('[') => {
                    let (_, open_brace_span) = lexer.next();
                    let index = self.parse_general_expression(lexer, ctx.reborrow())?;
                    let close_brace_span = lexer.expect_span(Token::Paren(']'))?;

                    // WGSL doesn't allow pointers to be subscripted. But Naga IR doesn't
                    // distinguish pointers and references, so we must check here.
                    if wgsl_pointer {
                        return Err(Error::Pointer(
                            "the value indexed by a `[]` subscripting expression",
                            prefix_span,
                        ));
                    }

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
                _ => break,
            };

            prefix_span = lexer.span_from(span_start);
            handle = ctx
                .expressions
                .append(expression, NagaSpan::from(prefix_span.clone()));
        }

        Ok(TypedExpression {
            handle,
            is_reference,
        })
    }

    /// Parse a `unary_expression`.
    fn parse_unary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<TypedExpression, Error<'a>> {
        self.push_scope(Scope::UnaryExpr, lexer);
        //TODO: refactor this to avoid backing up
        let expr = match lexer.peek().0 {
            Token::Operation('-') => {
                let _ = lexer.next();
                let unloaded_expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ctx.apply_load_rule(unloaded_expr);
                let expr = crate::Expression::Unary {
                    op: crate::UnaryOperator::Negate,
                    expr,
                };
                let span = NagaSpan::from(self.peek_scope(lexer));
                TypedExpression::non_reference(ctx.expressions.append(expr, span))
            }
            Token::Operation('!' | '~') => {
                let _ = lexer.next();
                let unloaded_expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ctx.apply_load_rule(unloaded_expr);
                let expr = crate::Expression::Unary {
                    op: crate::UnaryOperator::Not,
                    expr,
                };
                let span = NagaSpan::from(self.peek_scope(lexer));
                TypedExpression::non_reference(ctx.expressions.append(expr, span))
            }
            Token::Operation('*') => {
                let _ = lexer.next();
                // The `*` operator does not accept a reference, so we must apply the Load
                // Rule here. But the operator itself simply changes the type from
                // `ptr<SC, T, A>` to `ref<SC, T, A>`, so we generate no code for the
                // operator itself. We simply return a `TypedExpression` with
                // `is_reference` set to true.
                let unloaded_pointer = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let pointer = ctx.apply_load_rule(unloaded_pointer);

                // An expression like `&*ptr` may generate no Naga IR at all, but WGSL requires
                // an error if `ptr` is not a pointer. So we have to type-check this ourselves.
                if ctx.resolve_type(pointer)?.pointer_space().is_none() {
                    let span = ctx
                        .expressions
                        .get_span(pointer)
                        .to_range()
                        .unwrap_or_else(|| self.peek_scope(lexer));
                    return Err(Error::NotPointer(span));
                }

                TypedExpression {
                    handle: pointer,
                    is_reference: true,
                }
            }
            Token::Operation('&') => {
                let _ = lexer.next();
                // The `&` operator simply converts a reference to a pointer. And since a
                // reference is required, the Load Rule is not applied.
                let operand = self.parse_unary_expression(lexer, ctx.reborrow())?;
                if !operand.is_reference {
                    let span = ctx
                        .expressions
                        .get_span(operand.handle)
                        .to_range()
                        .unwrap_or_else(|| self.peek_scope(lexer));
                    return Err(Error::NotReference("the operand of the `&` operator", span));
                }

                // No code is generated. We just declare the pointer a reference now.
                TypedExpression {
                    is_reference: false,
                    ..operand
                }
            }
            _ => self.parse_singular_expression(lexer, ctx.reborrow())?,
        };

        self.pop_scope(lexer);
        Ok(expr)
    }

    /// Parse a `singular_expression`.
    fn parse_singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<TypedExpression, Error<'a>> {
        let start = lexer.current_byte_offset();
        self.push_scope(Scope::SingularExpr, lexer);
        let primary_expr = self.parse_primary_expression(lexer, ctx.reborrow())?;
        let singular_expr = self.parse_postfix(start, lexer, ctx.reborrow(), primary_expr)?;
        self.pop_scope(lexer);

        Ok(singular_expr)
    }

    fn parse_equality_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<TypedExpression, Error<'a>> {
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
                                                self.parse_unary_expression(lexer, context)
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

    fn parse_general_expression_with_span<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<(Handle<crate::Expression>, Span), Error<'a>> {
        let (expr, span) = self.parse_general_expression_for_reference(lexer, ctx.reborrow())?;
        Ok((ctx.apply_load_rule(expr), span))
    }

    fn parse_general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'a>> {
        let (expr, _span) = self.parse_general_expression_for_reference(lexer, ctx.reborrow())?;
        Ok(ctx.apply_load_rule(expr))
    }

    fn parse_general_expression_for_reference<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<(TypedExpression, Span), Error<'a>> {
        self.push_scope(Scope::GeneralExpr, lexer);
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
        Ok((handle, self.pop_scope(lexer)))
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(&'a str, Span, Handle<crate::Type>), Error<'a>> {
        let (name, name_span) = lexer.next_ident_with_span()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
        Ok((name, name_span, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<ParsedVariable<'a>, Error<'a>> {
        self.push_scope(Scope::VariableDecl, lexer);
        let mut space = None;

        if lexer.skip(Token::Paren('<')) {
            let (class_str, span) = lexer.next_ident_with_span()?;
            space = Some(match class_str {
                "storage" => {
                    let access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        // defaulting to `read`
                        crate::StorageAccess::LOAD
                    };
                    crate::AddressSpace::Storage { access }
                }
                _ => conv::map_address_space(class_str, span)?,
            });
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;

        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.parse_const_expression(lexer, type_arena, const_arena)?;
            Some(handle)
        } else {
            None
        };
        lexer.expect(Token::Separator(';'))?;
        let name_span = self.pop_scope(lexer);
        Ok(ParsedVariable {
            name,
            name_span,
            space,
            ty,
            init,
        })
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<(Vec<crate::StructMember>, u32), Error<'a>> {
        let mut offset = 0;
        let mut struct_alignment = Alignment::ONE;
        let mut members = Vec::new();

        lexer.expect(Token::Paren('{'))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren('}')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next(),
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let (mut size_attr, mut align_attr) = (None, None);
            self.push_scope(Scope::Attribute, lexer);
            let mut bind_parser = BindingParser::default();
            while lexer.skip(Token::Attribute) {
                match lexer.next_ident_with_span()? {
                    ("size", _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (value, span) =
                            lexer.capture_span(Self::parse_non_negative_i32_literal)?;
                        lexer.expect(Token::Paren(')'))?;
                        size_attr = Some((value, span));
                    }
                    ("align", _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (value, span) =
                            lexer.capture_span(Self::parse_non_negative_i32_literal)?;
                        lexer.expect(Token::Paren(')'))?;
                        align_attr = Some((value, span));
                    }
                    (word, word_span) => bind_parser.parse(lexer, word, word_span)?,
                }
            }

            let bind_span = self.pop_scope(lexer);
            let mut binding = bind_parser.finish(bind_span)?;

            let (name, span) = match lexer.next() {
                (Token::Word(word), span) => (word, span),
                other => return Err(Error::Unexpected(other, ExpectedToken::FieldName)),
            };
            if crate::keywords::wgsl::RESERVED.contains(&name) {
                return Err(Error::ReservedKeyword(span));
            }
            lexer.expect(Token::Separator(':'))?;
            let ty = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
            ready = lexer.skip(Token::Separator(','));

            self.layouter.update(type_arena, const_arena).unwrap();

            let member_min_size = self.layouter[ty].size;
            let member_min_alignment = self.layouter[ty].alignment;

            let member_size = if let Some((size, span)) = size_attr {
                if size < member_min_size {
                    return Err(Error::SizeAttributeTooLow(span, member_min_size));
                } else {
                    size
                }
            } else {
                member_min_size
            };

            let member_alignment = if let Some((align, span)) = align_attr {
                if let Some(alignment) = Alignment::new(align) {
                    if alignment < member_min_alignment {
                        return Err(Error::AlignAttributeTooLow(span, member_min_alignment));
                    } else {
                        alignment
                    }
                } else {
                    return Err(Error::NonPowerOfTwoAlignAttribute(span));
                }
            } else {
                member_min_alignment
            };

            offset = member_alignment.round_up(offset);
            struct_alignment = struct_alignment.max(member_alignment);

            if let Some(ref mut binding) = binding {
                binding.apply_default_interpolation(&type_arena[ty].inner);
            }

            members.push(crate::StructMember {
                name: Some(name.to_owned()),
                ty,
                binding,
                offset,
            });

            offset += member_size;
        }

        let struct_size = struct_alignment.round_up(offset);
        Ok((members, struct_size))
    }

    fn parse_matrix_scalar_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    ) -> Result<crate::TypeInner, Error<'a>> {
        let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
        match kind {
            crate::ScalarKind::Float => Ok(crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            }),
            _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
        }
    }

    fn parse_type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        _attribute: TypeAttributes,
        word: &'a str,
        type_arena: &mut UniqueArena<crate::Type>,
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
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Bi)?
            }
            "mat2x3" => {
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Tri)?
            }
            "mat2x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Bi,
                crate::VectorSize::Quad,
            )?,
            "mat3x2" => {
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Tri, crate::VectorSize::Bi)?
            }
            "mat3x3" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Tri,
                crate::VectorSize::Tri,
            )?,
            "mat3x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Tri,
                crate::VectorSize::Quad,
            )?,
            "mat4x2" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Bi,
            )?,
            "mat4x3" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Tri,
            )?,
            "mat4x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Quad,
            )?,
            "atomic" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                crate::TypeInner::Atomic { kind, width }
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let (ident, span) = lexer.next_ident_with_span()?;
                let mut space = conv::map_address_space(ident, span)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                if let crate::AddressSpace::Storage { ref mut access } = space {
                    *access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        crate::StorageAccess::LOAD
                    };
                }
                lexer.expect_generic_paren('>')?;
                crate::TypeInner::Pointer { base, space }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let const_handle =
                        self.parse_const_expression(lexer, type_arena, const_arena)?;
                    crate::ArraySize::Constant(const_handle)
                } else {
                    crate::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                let stride = {
                    self.layouter.update(type_arena, const_arena).unwrap();
                    self.layouter[base].to_stride()
                };
                crate::TypeInner::Array { base, size, stride }
            }
            "binding_array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, None, type_arena, const_arena)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let const_handle =
                        self.parse_const_expression(lexer, type_arena, const_arena)?;
                    crate::ArraySize::Constant(const_handle)
                } else {
                    crate::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                crate::TypeInner::BindingArray { base, size }
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
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_2d_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube_array" => crate::TypeInner::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_multisampled_2d" => crate::TypeInner::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: true },
            },
            "texture_storage_1d" => {
                let (format, access) = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_1d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d" => {
                let (format, access) = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_3d" => {
                let (format, access) = lexer.next_format_generic()?;
                crate::TypeInner::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            _ => return Ok(None),
        }))
    }

    const fn check_texture_sample_type(
        kind: crate::ScalarKind,
        width: u8,
        span: Span,
    ) -> Result<(), Error<'static>> {
        use crate::ScalarKind::*;
        // Validate according to https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
        match (kind, width) {
            (Float | Sint | Uint, 4) => Ok(()),
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
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        Ok(match self.lookup_type.get(name) {
            Some(&handle) => handle,
            None => {
                match self.parse_type_decl_impl(lexer, attribute, name, type_arena, const_arena)? {
                    Some(inner) => {
                        let span = name_span.start..lexer.current_byte_offset();
                        type_arena.insert(
                            crate::Type {
                                name: debug_name.map(|s| s.to_string()),
                                inner,
                            },
                            NagaSpan::from(span),
                        )
                    }
                    None => return Err(Error::UnknownType(name_span)),
                }
            }
        })
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        debug_name: Option<&'a str>,
        type_arena: &mut UniqueArena<crate::Type>,
        const_arena: &mut Arena<crate::Constant>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        self.push_scope(Scope::TypeDecl, lexer);
        let attribute = TypeAttributes::default();

        if lexer.skip(Token::Attribute) {
            let other = lexer.next();
            return Err(Error::Unexpected(other, ExpectedToken::TypeAttribute));
        }

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
        self.pop_scope(lexer);
        // Only set span if it's the first occurrence of the type.
        // Type spans therefore should only be used for errors in type declarations;
        // use variable spans/expression spans/etc. otherwise
        Ok(handle)
    }

    /// Parse an assignment statement (will also parse increment and decrement statements)
    fn parse_assignment_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, 'out>,
    ) -> Result<(), Error<'a>> {
        use crate::BinaryOperator as Bo;

        let span_start = lexer.current_byte_offset();
        context.emitter.start(context.expressions);
        let reference = self.parse_unary_expression(lexer, context.reborrow())?;
        // The left hand side of an assignment must be a reference.
        let lhs_span = span_start..lexer.current_byte_offset();
        if !reference.is_reference {
            return Err(Error::NotReference(
                "the left-hand side of an assignment",
                lhs_span,
            ));
        }

        let value = match lexer.next() {
            (Token::Operation('='), _) => {
                self.parse_general_expression(lexer, context.reborrow())?
            }
            (Token::AssignmentOperation(c), span) => {
                let op = match c {
                    '<' => Bo::ShiftLeft,
                    '>' => Bo::ShiftRight,
                    '+' => Bo::Add,
                    '-' => Bo::Subtract,
                    '*' => Bo::Multiply,
                    '/' => Bo::Divide,
                    '%' => Bo::Modulo,
                    '&' => Bo::And,
                    '|' => Bo::InclusiveOr,
                    '^' => Bo::ExclusiveOr,
                    //Note: `consume_token` shouldn't produce any other assignment ops
                    _ => unreachable!(),
                };
                let left = context.expressions.append(
                    crate::Expression::Load {
                        pointer: reference.handle,
                    },
                    lhs_span.into(),
                );
                let right = self.parse_general_expression(lexer, context.reborrow())?;
                context
                    .expressions
                    .append(crate::Expression::Binary { op, left, right }, span.into())
            }
            token @ (Token::IncrementOperation | Token::DecrementOperation, _) => {
                let op = match token.0 {
                    Token::IncrementOperation => Bo::Add,
                    Token::DecrementOperation => Bo::Subtract,
                    _ => unreachable!(),
                };
                let op_span = token.1;

                // prepare the typifier, but work around mutable borrowing...
                let _ = context.resolve_type(reference.handle)?;

                let ty = context.typifier.get(reference.handle, context.types);
                let (kind, width) = match *ty {
                    crate::TypeInner::ValuePointer {
                        size: None,
                        kind,
                        width,
                        ..
                    } => (kind, width),
                    crate::TypeInner::Pointer { base, .. } => match context.types[base].inner {
                        crate::TypeInner::Scalar { kind, width } => (kind, width),
                        _ => return Err(Error::BadIncrDecrReferenceType(lhs_span)),
                    },
                    _ => return Err(Error::BadIncrDecrReferenceType(lhs_span)),
                };
                let constant_inner = crate::ConstantInner::Scalar {
                    width,
                    value: match kind {
                        crate::ScalarKind::Sint => crate::ScalarValue::Sint(1),
                        crate::ScalarKind::Uint => crate::ScalarValue::Uint(1),
                        _ => return Err(Error::BadIncrDecrReferenceType(lhs_span)),
                    },
                };
                let constant = context.constants.append(
                    crate::Constant {
                        name: None,
                        specialization: None,
                        inner: constant_inner,
                    },
                    crate::Span::default(),
                );

                let left = context.expressions.append(
                    crate::Expression::Load {
                        pointer: reference.handle,
                    },
                    lhs_span.into(),
                );
                let right = context.interrupt_emitter(
                    crate::Expression::Constant(constant),
                    crate::Span::default(),
                );
                context.expressions.append(
                    crate::Expression::Binary { op, left, right },
                    op_span.into(),
                )
            }
            other => return Err(Error::Unexpected(other, ExpectedToken::SwitchItem)),
        };

        let span_end = lexer.current_byte_offset();
        context
            .block
            .extend(context.emitter.finish(context.expressions));
        context.block.push(
            crate::Statement::Store {
                pointer: reference.handle,
                value,
            },
            NagaSpan::from(span_start..span_end),
        );
        Ok(())
    }

    /// Parse a function call statement.
    fn parse_function_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ident: &'a str,
        mut context: ExpressionContext<'a, '_, 'out>,
    ) -> Result<(), Error<'a>> {
        self.push_scope(Scope::SingularExpr, lexer);
        context.emitter.start(context.expressions);
        if self
            .parse_function_call_inner(lexer, ident, context.reborrow())?
            .is_none()
        {
            let span = lexer.next().1;
            return Err(Error::UnknownLocalFunction(span));
        }
        context
            .block
            .extend(context.emitter.finish(context.expressions));
        self.pop_scope(lexer);

        Ok(())
    }

    fn parse_switch_case_body<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, 'out>,
    ) -> Result<(bool, crate::Block), Error<'a>> {
        let mut body = crate::Block::new();
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

        Ok((fall_through, body))
    }

    fn parse_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, 'out>,
        block: &'out mut crate::Block,
        is_uniform_control_flow: bool,
    ) -> Result<(), Error<'a>> {
        self.push_scope(Scope::Statement, lexer);
        match lexer.peek() {
            (Token::Separator(';'), _) => {
                let _ = lexer.next();
                self.pop_scope(lexer);
                return Ok(());
            }
            (Token::Paren('{'), _) => {
                self.push_scope(Scope::Block, lexer);
                let _ = lexer.next();
                let mut statements = crate::Block::new();
                while !lexer.skip(Token::Paren('}')) {
                    self.parse_statement(
                        lexer,
                        context.reborrow(),
                        &mut statements,
                        is_uniform_control_flow,
                    )?;
                }
                self.pop_scope(lexer);
                let span = NagaSpan::from(self.pop_scope(lexer));
                block.push(crate::Statement::Block(statements), span);
                return Ok(());
            }
            (Token::Word(word), _) => {
                let mut emitter = super::Emitter::default();
                let statement = match word {
                    "_" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        lexer.expect(Token::Operation('='))?;
                        self.parse_general_expression(
                            lexer,
                            context.as_expression(block, &mut emitter),
                        )?;
                        lexer.expect(Token::Separator(';'))?;
                        block.extend(emitter.finish(context.expressions));
                        None
                    }
                    "let" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        let (name, name_span) = lexer.next_ident_with_span()?;
                        if crate::keywords::wgsl::RESERVED.contains(&name) {
                            return Err(Error::ReservedKeyword(name_span));
                        }
                        let given_ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.parse_type_decl(
                                lexer,
                                None,
                                context.types,
                                context.constants,
                            )?;
                            Some(ty)
                        } else {
                            None
                        };
                        lexer.expect(Token::Operation('='))?;
                        let expr_id = self.parse_general_expression(
                            lexer,
                            context.as_expression(block, &mut emitter),
                        )?;
                        lexer.expect(Token::Separator(';'))?;
                        if let Some(ty) = given_ty {
                            // prepare the typifier, but work around mutable borrowing...
                            let _ = context
                                .as_expression(block, &mut emitter)
                                .resolve_type(expr_id)?;
                            let expr_inner = context.typifier.get(expr_id, context.types);
                            let given_inner = &context.types[ty].inner;
                            if !given_inner.equivalent(expr_inner, context.types) {
                                log::error!(
                                    "Given type {:?} doesn't match expected {:?}",
                                    given_inner,
                                    expr_inner
                                );
                                return Err(Error::InitializationTypeMismatch(
                                    name_span,
                                    expr_inner.to_wgsl(context.types, context.constants),
                                ));
                            }
                        }
                        block.extend(emitter.finish(context.expressions));
                        context.lookup_ident.insert(
                            name,
                            TypedExpression {
                                handle: expr_id,
                                is_reference: false,
                            },
                        );
                        context
                            .named_expressions
                            .insert(expr_id, String::from(name));
                        None
                    }
                    "var" => {
                        let _ = lexer.next();
                        enum Init {
                            Empty,
                            Constant(Handle<crate::Constant>),
                            Variable(Handle<crate::Expression>),
                        }

                        let (name, name_span) = lexer.next_ident_with_span()?;
                        if crate::keywords::wgsl::RESERVED.contains(&name) {
                            return Err(Error::ReservedKeyword(name_span));
                        }
                        let given_ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.parse_type_decl(
                                lexer,
                                None,
                                context.types,
                                context.constants,
                            )?;
                            Some(ty)
                        } else {
                            None
                        };

                        let (init, ty) = if lexer.skip(Token::Operation('=')) {
                            emitter.start(context.expressions);
                            let value = self.parse_general_expression(
                                lexer,
                                context.as_expression(block, &mut emitter),
                            )?;
                            block.extend(emitter.finish(context.expressions));

                            // prepare the typifier, but work around mutable borrowing...
                            let _ = context
                                .as_expression(block, &mut emitter)
                                .resolve_type(value)?;

                            //TODO: share more of this code with `let` arm
                            let ty = match given_ty {
                                Some(ty) => {
                                    let expr_inner = context.typifier.get(value, context.types);
                                    let given_inner = &context.types[ty].inner;
                                    if !given_inner.equivalent(expr_inner, context.types) {
                                        log::error!(
                                            "Given type {:?} doesn't match expected {:?}",
                                            given_inner,
                                            expr_inner
                                        );
                                        return Err(Error::InitializationTypeMismatch(
                                            name_span,
                                            expr_inner.to_wgsl(context.types, context.constants),
                                        ));
                                    }
                                    ty
                                }
                                None => {
                                    // register the type, if needed
                                    match context.typifier[value].clone() {
                                        TypeResolution::Handle(ty) => ty,
                                        TypeResolution::Value(inner) => context.types.insert(
                                            crate::Type { name: None, inner },
                                            Default::default(),
                                        ),
                                    }
                                }
                            };

                            let init = match context.expressions[value] {
                                crate::Expression::Constant(handle) if is_uniform_control_flow => {
                                    Init::Constant(handle)
                                }
                                _ => Init::Variable(value),
                            };
                            (init, ty)
                        } else {
                            match given_ty {
                                Some(ty) => (Init::Empty, ty),
                                None => {
                                    log::error!(
                                        "Variable '{}' without an initializer needs a type",
                                        name
                                    );
                                    return Err(Error::MissingType(name_span));
                                }
                            }
                        };

                        lexer.expect(Token::Separator(';'))?;
                        let var_id = context.variables.append(
                            crate::LocalVariable {
                                name: Some(name.to_owned()),
                                ty,
                                init: match init {
                                    Init::Constant(value) => Some(value),
                                    _ => None,
                                },
                            },
                            NagaSpan::from(name_span),
                        );

                        // Doesn't make sense to assign a span to cached lookup
                        let expr_id = context
                            .expressions
                            .append(crate::Expression::LocalVariable(var_id), Default::default());
                        context.lookup_ident.insert(
                            name,
                            TypedExpression {
                                handle: expr_id,
                                is_reference: true,
                            },
                        );

                        if let Init::Variable(value) = init {
                            Some(crate::Statement::Store {
                                pointer: expr_id,
                                value,
                            })
                        } else {
                            None
                        }
                    }
                    "return" => {
                        let _ = lexer.next();
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
                        Some(crate::Statement::Return { value })
                    }
                    "if" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        let condition = self.parse_general_expression(
                            lexer,
                            context.as_expression(block, &mut emitter),
                        )?;
                        block.extend(emitter.finish(context.expressions));

                        let accept = self.parse_block(lexer, context.reborrow(), false)?;

                        let mut elsif_stack = Vec::new();
                        let mut elseif_span_start = lexer.current_byte_offset();
                        let mut reject = loop {
                            if !lexer.skip(Token::Word("else")) {
                                break crate::Block::new();
                            }

                            if !lexer.skip(Token::Word("if")) {
                                // ... else { ... }
                                break self.parse_block(lexer, context.reborrow(), false)?;
                            }

                            // ... else if (...) { ... }
                            let mut sub_emitter = super::Emitter::default();

                            sub_emitter.start(context.expressions);
                            let other_condition = self.parse_general_expression(
                                lexer,
                                context.as_expression(block, &mut sub_emitter),
                            )?;
                            let other_emit = sub_emitter.finish(context.expressions);
                            let other_block = self.parse_block(lexer, context.reborrow(), false)?;
                            elsif_stack.push((
                                elseif_span_start,
                                other_condition,
                                other_emit,
                                other_block,
                            ));
                            elseif_span_start = lexer.current_byte_offset();
                        };

                        let span_end = lexer.current_byte_offset();
                        // reverse-fold the else-if blocks
                        //Note: we may consider uplifting this to the IR
                        for (other_span_start, other_cond, other_emit, other_block) in
                            elsif_stack.into_iter().rev()
                        {
                            let sub_stmt = crate::Statement::If {
                                condition: other_cond,
                                accept: other_block,
                                reject,
                            };
                            reject = crate::Block::new();
                            reject.extend(other_emit);
                            reject.push(sub_stmt, NagaSpan::from(other_span_start..span_end))
                        }

                        Some(crate::Statement::If {
                            condition,
                            accept,
                            reject,
                        })
                    }
                    "switch" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        let selector = self.parse_general_expression(
                            lexer,
                            context.as_expression(block, &mut emitter),
                        )?;
                        let uint = Some(crate::ScalarKind::Uint)
                            == context
                                .as_expression(block, &mut emitter)
                                .resolve_type(selector)?
                                .scalar_kind();
                        block.extend(emitter.finish(context.expressions));
                        lexer.expect(Token::Paren('{'))?;
                        let mut cases = Vec::new();

                        loop {
                            // cases + default
                            match lexer.next() {
                                (Token::Word("case"), _) => {
                                    // parse a list of values
                                    let value = loop {
                                        let value = Self::parse_switch_value(lexer, uint)?;
                                        if lexer.skip(Token::Separator(',')) {
                                            if lexer.skip(Token::Separator(':')) {
                                                break value;
                                            }
                                        } else {
                                            lexer.skip(Token::Separator(':'));
                                            break value;
                                        }
                                        cases.push(crate::SwitchCase {
                                            value: crate::SwitchValue::Integer(value),
                                            body: crate::Block::new(),
                                            fall_through: true,
                                        });
                                    };

                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, context.reborrow())?;

                                    cases.push(crate::SwitchCase {
                                        value: crate::SwitchValue::Integer(value),
                                        body,
                                        fall_through,
                                    });
                                }
                                (Token::Word("default"), _) => {
                                    lexer.skip(Token::Separator(':'));
                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, context.reborrow())?;
                                    cases.push(crate::SwitchCase {
                                        value: crate::SwitchValue::Default,
                                        body,
                                        fall_through,
                                    });
                                }
                                (Token::Paren('}'), _) => break,
                                other => {
                                    return Err(Error::Unexpected(other, ExpectedToken::SwitchItem))
                                }
                            }
                        }

                        Some(crate::Statement::Switch { selector, cases })
                    }
                    "loop" => Some(self.parse_loop(lexer, context.reborrow(), &mut emitter)?),
                    "while" => {
                        let _ = lexer.next();
                        let mut body = crate::Block::new();

                        let (condition, span) = lexer.capture_span(|lexer| {
                            emitter.start(context.expressions);
                            let condition = self.parse_general_expression(
                                lexer,
                                context.as_expression(&mut body, &mut emitter),
                            )?;
                            lexer.expect(Token::Paren('{'))?;
                            body.extend(emitter.finish(context.expressions));
                            Ok(condition)
                        })?;
                        let mut reject = crate::Block::new();
                        reject.push(crate::Statement::Break, NagaSpan::default());
                        body.push(
                            crate::Statement::If {
                                condition,
                                accept: crate::Block::new(),
                                reject,
                            },
                            NagaSpan::from(span),
                        );

                        while !lexer.skip(Token::Paren('}')) {
                            self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
                        }

                        Some(crate::Statement::Loop {
                            body,
                            continuing: crate::Block::new(),
                            break_if: None,
                        })
                    }
                    "for" => {
                        let _ = lexer.next();
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
                                    crate::Statement::Store { .. }
                                    | crate::Statement::Call { .. } => {}
                                    _ => return Err(Error::InvalidForInitializer(span)),
                                }
                            }
                        };

                        let mut body = crate::Block::new();
                        if !lexer.skip(Token::Separator(';')) {
                            let (condition, span) = lexer.capture_span(|lexer| {
                                emitter.start(context.expressions);
                                let condition = self.parse_general_expression(
                                    lexer,
                                    context.as_expression(&mut body, &mut emitter),
                                )?;
                                lexer.expect(Token::Separator(';'))?;
                                body.extend(emitter.finish(context.expressions));
                                Ok(condition)
                            })?;
                            let mut reject = crate::Block::new();
                            reject.push(crate::Statement::Break, NagaSpan::default());
                            body.push(
                                crate::Statement::If {
                                    condition,
                                    accept: crate::Block::new(),
                                    reject,
                                },
                                NagaSpan::from(span),
                            );
                        };

                        let mut continuing = crate::Block::new();
                        if !lexer.skip(Token::Paren(')')) {
                            match lexer.peek().0 {
                                Token::Word(ident) if context.lookup_ident.get(ident).is_none() => {
                                    self.parse_function_statement(
                                        lexer,
                                        ident,
                                        context.as_expression(&mut continuing, &mut emitter),
                                    )?
                                }
                                _ => self.parse_assignment_statement(
                                    lexer,
                                    context.as_expression(&mut continuing, &mut emitter),
                                )?,
                            }
                            lexer.expect(Token::Paren(')'))?;
                        }
                        lexer.expect(Token::Paren('{'))?;

                        while !lexer.skip(Token::Paren('}')) {
                            self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
                        }

                        Some(crate::Statement::Loop {
                            body,
                            continuing,
                            break_if: None,
                        })
                    }
                    "break" => {
                        let (_, mut span) = lexer.next();
                        // Check if the next token is an `if`, this indicates
                        // that the user tried to type out a `break if` which
                        // is illegal in this position.
                        let (peeked_token, peeked_span) = lexer.peek();
                        if let Token::Word("if") = peeked_token {
                            span.end = peeked_span.end;
                            return Err(Error::InvalidBreakIf(span));
                        }
                        Some(crate::Statement::Break)
                    }
                    "continue" => {
                        let _ = lexer.next();
                        Some(crate::Statement::Continue)
                    }
                    "discard" => {
                        let _ = lexer.next();
                        Some(crate::Statement::Kill)
                    }
                    "storageBarrier" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Paren('('))?;
                        lexer.expect(Token::Paren(')'))?;
                        Some(crate::Statement::Barrier(crate::Barrier::STORAGE))
                    }
                    "workgroupBarrier" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Paren('('))?;
                        lexer.expect(Token::Paren(')'))?;
                        Some(crate::Statement::Barrier(crate::Barrier::WORK_GROUP))
                    }
                    "atomicStore" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        lexer.open_arguments()?;
                        let mut expression_ctx = context.as_expression(block, &mut emitter);
                        let pointer =
                            self.parse_atomic_pointer(lexer, expression_ctx.reborrow())?;
                        lexer.expect(Token::Separator(','))?;
                        let value = self.parse_general_expression(lexer, expression_ctx)?;
                        lexer.close_arguments()?;
                        block.extend(emitter.finish(context.expressions));
                        Some(crate::Statement::Store { pointer, value })
                    }
                    "textureStore" => {
                        let _ = lexer.next();
                        emitter.start(context.expressions);
                        lexer.open_arguments()?;
                        let mut expr_context = context.as_expression(block, &mut emitter);
                        let (image, image_span) = self
                            .parse_general_expression_with_span(lexer, expr_context.reborrow())?;
                        lexer.expect(Token::Separator(','))?;
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
                        let value = self.parse_general_expression(
                            lexer,
                            context.as_expression(block, &mut emitter),
                        )?;
                        lexer.close_arguments()?;
                        block.extend(emitter.finish(context.expressions));
                        Some(crate::Statement::ImageStore {
                            image,
                            coordinate,
                            array_index,
                            value,
                        })
                    }
                    // assignment or a function call
                    ident => {
                        match context.lookup_ident.get(ident) {
                            Some(_) => self.parse_assignment_statement(
                                lexer,
                                context.as_expression(block, &mut emitter),
                            )?,
                            None => self.parse_function_statement(
                                lexer,
                                ident,
                                context.as_expression(block, &mut emitter),
                            )?,
                        }
                        lexer.expect(Token::Separator(';'))?;
                        None
                    }
                };
                let span = NagaSpan::from(self.pop_scope(lexer));
                if let Some(statement) = statement {
                    block.push(statement, span);
                }
            }
            _ => {
                let mut emitter = super::Emitter::default();
                self.parse_assignment_statement(lexer, context.as_expression(block, &mut emitter))?;
                self.pop_scope(lexer);
            }
        }
        Ok(())
    }

    fn parse_loop<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
        emitter: &mut super::Emitter,
    ) -> Result<crate::Statement, Error<'a>> {
        let _ = lexer.next();
        let mut body = crate::Block::new();
        let mut continuing = crate::Block::new();
        let mut break_if = None;
        lexer.expect(Token::Paren('{'))?;

        loop {
            if lexer.skip(Token::Word("continuing")) {
                // Branch for the `continuing` block, this must be
                // the last thing in the loop body

                // Expect a opening brace to start the continuing block
                lexer.expect(Token::Paren('{'))?;
                loop {
                    if lexer.skip(Token::Word("break")) {
                        // Branch for the `break if` statement, this statement
                        // has the form `break if <expr>;` and must be the last
                        // statement in a continuing block

                        // The break must be followed by an `if` to form
                        // the break if
                        lexer.expect(Token::Word("if"))?;

                        // Start the emitter to begin parsing an expression
                        emitter.start(context.expressions);
                        let condition = self.parse_general_expression(
                            lexer,
                            context.as_expression(&mut body, emitter),
                        )?;
                        // Add all emits to the continuing body
                        continuing.extend(emitter.finish(context.expressions));
                        // Set the condition of the break if to the newly parsed
                        // expression
                        break_if = Some(condition);

                        // Expext a semicolon to close the statement
                        lexer.expect(Token::Separator(';'))?;
                        // Expect a closing brace to close the continuing block,
                        // since the break if must be the last statement
                        lexer.expect(Token::Paren('}'))?;
                        // Stop parsing the continuing block
                        break;
                    } else if lexer.skip(Token::Paren('}')) {
                        // If we encounter a closing brace it means we have reached
                        // the end of the continuing block and should stop processing
                        break;
                    } else {
                        // Otherwise try to parse a statement
                        self.parse_statement(lexer, context.reborrow(), &mut continuing, false)?;
                    }
                }
                // Since the continuing block must be the last part of the loop body,
                // we expect to see a closing brace to end the loop body
                lexer.expect(Token::Paren('}'))?;
                break;
            }
            if lexer.skip(Token::Paren('}')) {
                // If we encounter a closing brace it means we have reached
                // the end of the loop body and should stop processing
                break;
            }
            // Otherwise try to parse a statement
            self.parse_statement(lexer, context.reborrow(), &mut body, false)?;
        }

        Ok(crate::Statement::Loop {
            body,
            continuing,
            break_if,
        })
    }

    fn parse_block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: StatementContext<'a, '_, '_>,
        is_uniform_control_flow: bool,
    ) -> Result<crate::Block, Error<'a>> {
        self.push_scope(Scope::Block, lexer);
        lexer.expect(Token::Paren('{'))?;
        let mut block = crate::Block::new();
        while !lexer.skip(Token::Paren('}')) {
            self.parse_statement(
                lexer,
                context.reborrow(),
                &mut block,
                is_uniform_control_flow,
            )?;
        }
        self.pop_scope(lexer);
        Ok(block)
    }

    fn parse_varying_binding<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
    ) -> Result<Option<crate::Binding>, Error<'a>> {
        let mut bind_parser = BindingParser::default();
        self.push_scope(Scope::Attribute, lexer);

        while lexer.skip(Token::Attribute) {
            let (word, span) = lexer.next_ident_with_span()?;
            bind_parser.parse(lexer, word, span)?;
        }

        let span = self.pop_scope(lexer);
        bind_parser.finish(span)
    }

    fn parse_function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        module: &mut crate::Module,
        lookup_global_expression: &FastHashMap<&'a str, crate::Expression>,
    ) -> Result<(crate::Function, &'a str), Error<'a>> {
        self.push_scope(Scope::FunctionDecl, lexer);
        // read function name
        let mut lookup_ident = FastHashMap::default();
        let (fun_name, span) = lexer.next_ident_with_span()?;
        if crate::keywords::wgsl::RESERVED.contains(&fun_name) {
            return Err(Error::ReservedKeyword(span));
        }
        if let Some(entry) = self
            .module_scope_identifiers
            .insert(String::from(fun_name), span.clone())
        {
            return Err(Error::Redefinition {
                previous: entry,
                current: span,
            });
        }
        // populate initial expressions
        let mut expressions = Arena::new();
        for (&name, expression) in lookup_global_expression.iter() {
            let (span, is_reference) = match *expression {
                crate::Expression::GlobalVariable(handle) => (
                    module.global_variables.get_span(handle),
                    module.global_variables[handle].space != crate::AddressSpace::Handle,
                ),
                crate::Expression::Constant(handle) => (module.constants.get_span(handle), false),
                _ => unreachable!(),
            };
            let expression = expressions.append(expression.clone(), span);
            lookup_ident.insert(
                name,
                TypedExpression {
                    handle: expression,
                    is_reference,
                },
            );
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
            let mut binding = self.parse_varying_binding(lexer)?;
            let (param_name, param_name_span, param_type) =
                self.parse_variable_ident_decl(lexer, &mut module.types, &mut module.constants)?;
            if crate::keywords::wgsl::RESERVED.contains(&param_name) {
                return Err(Error::ReservedKeyword(param_name_span));
            }
            let param_index = arguments.len() as u32;
            let expression = expressions.append(
                crate::Expression::FunctionArgument(param_index),
                NagaSpan::from(param_name_span),
            );
            lookup_ident.insert(
                param_name,
                TypedExpression {
                    handle: expression,
                    is_reference: false,
                },
            );
            if let Some(ref mut binding) = binding {
                binding.apply_default_interpolation(&module.types[param_type].inner);
            }
            arguments.push(crate::FunctionArgument {
                name: Some(param_name.to_string()),
                ty: param_type,
                binding,
            });
            ready = lexer.skip(Token::Separator(','));
        }
        // read return type
        let result = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let mut binding = self.parse_varying_binding(lexer)?;
            let ty = self.parse_type_decl(lexer, None, &mut module.types, &mut module.constants)?;
            if let Some(ref mut binding) = binding {
                binding.apply_default_interpolation(&module.types[ty].inner);
            }
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
            body: crate::Block::new(),
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
        self.pop_scope(lexer);

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
        let mut stage = None;
        let mut workgroup_size = [0u32; 3];
        let mut early_depth_test = None;
        let (mut bind_index, mut bind_group) = (None, None);

        self.push_scope(Scope::Attribute, lexer);
        while lexer.skip(Token::Attribute) {
            match lexer.next_ident_with_span()? {
                ("binding", _) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_index = Some(Self::parse_non_negative_i32_literal(lexer)?);
                    lexer.expect(Token::Paren(')'))?;
                }
                ("group", _) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_group = Some(Self::parse_non_negative_i32_literal(lexer)?);
                    lexer.expect(Token::Paren(')'))?;
                }
                ("vertex", _) => {
                    stage = Some(crate::ShaderStage::Vertex);
                }
                ("fragment", _) => {
                    stage = Some(crate::ShaderStage::Fragment);
                }
                ("compute", _) => {
                    stage = Some(crate::ShaderStage::Compute);
                }
                ("workgroup_size", _) => {
                    lexer.expect(Token::Paren('('))?;
                    workgroup_size = [1u32; 3];
                    for (i, size) in workgroup_size.iter_mut().enumerate() {
                        *size = Self::parse_generic_non_negative_int_literal(lexer)?;
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
        }

        let attrib_scope = self.pop_scope(lexer);
        match (bind_group, bind_index) {
            (Some(group), Some(index)) => {
                binding = Some(crate::ResourceBinding {
                    group,
                    binding: index,
                });
            }
            (Some(_), None) => return Err(Error::MissingAttribute("binding", attrib_scope)),
            (None, Some(_)) => return Err(Error::MissingAttribute("group", attrib_scope)),
            (None, None) => {}
        }

        // read items
        let start = lexer.current_byte_offset();
        match lexer.next() {
            (Token::Separator(';'), _) => {}
            (Token::Word("struct"), _) => {
                let (name, span) = lexer.next_ident_with_span()?;
                if crate::keywords::wgsl::RESERVED.contains(&name) {
                    return Err(Error::ReservedKeyword(span));
                }
                let (members, span) =
                    self.parse_struct_body(lexer, &mut module.types, &mut module.constants)?;
                let type_span = NagaSpan::from(lexer.span_from(start));
                let ty = module.types.insert(
                    crate::Type {
                        name: Some(name.to_string()),
                        inner: crate::TypeInner::Struct { members, span },
                    },
                    type_span,
                );
                self.lookup_type.insert(name.to_owned(), ty);
            }
            (Token::Word("type"), _) => {
                let name = lexer.next_ident()?;
                lexer.expect(Token::Operation('='))?;
                let ty = self.parse_type_decl(
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;
                self.lookup_type.insert(name.to_owned(), ty);
                lexer.expect(Token::Separator(';'))?;
            }
            (Token::Word("let"), _) => {
                let (name, name_span) = lexer.next_ident_with_span()?;
                if crate::keywords::wgsl::RESERVED.contains(&name) {
                    return Err(Error::ReservedKeyword(name_span));
                }
                if let Some(entry) = self
                    .module_scope_identifiers
                    .insert(String::from(name), name_span.clone())
                {
                    return Err(Error::Redefinition {
                        previous: entry,
                        current: name_span,
                    });
                }
                let given_ty = if lexer.skip(Token::Separator(':')) {
                    let ty = self.parse_type_decl(
                        lexer,
                        None,
                        &mut module.types,
                        &mut module.constants,
                    )?;
                    Some(ty)
                } else {
                    None
                };

                lexer.expect(Token::Operation('='))?;
                let first_token_span = lexer.next();
                let const_handle = self.parse_const_expression_impl(
                    first_token_span,
                    lexer,
                    Some(name),
                    &mut module.types,
                    &mut module.constants,
                )?;

                if let Some(explicit_ty) = given_ty {
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
                        let expected_inner_str = match con.inner {
                            crate::ConstantInner::Scalar { width, value } => {
                                crate::TypeInner::Scalar {
                                    kind: value.scalar_kind(),
                                    width,
                                }
                                .to_wgsl(&module.types, &module.constants)
                            }
                            crate::ConstantInner::Composite { .. } => module.types[explicit_ty]
                                .inner
                                .to_wgsl(&module.types, &module.constants),
                        };
                        return Err(Error::InitializationTypeMismatch(
                            name_span,
                            expected_inner_str,
                        ));
                    }
                }

                lexer.expect(Token::Separator(';'))?;
                lookup_global_expression.insert(name, crate::Expression::Constant(const_handle));
            }
            (Token::Word("var"), _) => {
                let pvar =
                    self.parse_variable_decl(lexer, &mut module.types, &mut module.constants)?;
                if crate::keywords::wgsl::RESERVED.contains(&pvar.name) {
                    return Err(Error::ReservedKeyword(pvar.name_span));
                }
                if let Some(entry) = self
                    .module_scope_identifiers
                    .insert(String::from(pvar.name), pvar.name_span.clone())
                {
                    return Err(Error::Redefinition {
                        previous: entry,
                        current: pvar.name_span,
                    });
                }
                let var_handle = module.global_variables.append(
                    crate::GlobalVariable {
                        name: Some(pvar.name.to_owned()),
                        space: pvar.space.unwrap_or(crate::AddressSpace::Handle),
                        binding: binding.take(),
                        ty: pvar.ty,
                        init: pvar.init,
                    },
                    NagaSpan::from(pvar.name_span),
                );
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
                        module
                            .functions
                            .append(function, NagaSpan::from(lexer.span_from(start)));
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
        self.reset();

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
    pub const fn new() -> Self {
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
