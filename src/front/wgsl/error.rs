use crate::front::wgsl::parse::lexer::Token;
use crate::proc::{Alignment, ResolveError};
use crate::{SourceLocation, Span};
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use std::borrow::Cow;
use std::ops::Range;
use termcolor::{ColorChoice, NoColor, StandardStream};
use thiserror::Error;

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
            .map(|&(span, ref msg)| (span, msg.as_ref()))
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
                        Label::primary((), label.0.to_range().unwrap())
                            .with_message(label.1.to_string())
                    })
                    .collect(),
            )
            .with_notes(
                self.notes
                    .iter()
                    .map(|note| format!("note: {note}"))
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
        self.emit_to_string_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string_with_path(&self, source: &str, path: &str) -> String {
        let files = SimpleFile::new(path, source);
        let config = codespan_reporting::term::Config::default();
        let mut writer = NoColor::new(Vec::new());
        term::emit(&mut writer, &config, &files, &self.diagnostic()).expect("cannot write error");
        String::from_utf8(writer.into_inner()).unwrap()
    }

    /// Returns a [`SourceLocation`] for the first label in the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        self.labels.get(0).map(|label| label.0.location(source))
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    Number,
    Integer,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: assignment, increment/decrement expression
    Assignment,
    /// Expected: 'case', 'default', '}'
    SwitchItem,
    /// Expected: ',', ')'
    WorkgroupSizeSeparator,
    /// Expected: 'struct', 'let', 'var', 'type', ';', 'fn', eof
    GlobalItem,
    /// Expected a type.
    Type,
    /// Access of `var`, `let`, `const`.
    Variable,
    /// Access of a function
    Function,
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InvalidAssignmentType {
    Other,
    Swizzle,
    ImmutableBinding(Span),
}

#[derive(Clone, Debug)]
pub enum Error<'a> {
    Unexpected(Span, ExpectedToken<'a>),
    UnexpectedComponents(Span),
    UnexpectedOperationInConstContext(Span),
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
    InvalidGatherComponent(Span),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    SizeAttributeTooLow(Span, u32),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    TypeNotConstructible(Span),
    TypeNotInferrable(Span),
    InitializationTypeMismatch(Span, String, String),
    MissingType(Span),
    MissingAttribute(&'static str, Span),
    InvalidAtomicPointer(Span),
    InvalidAtomicOperandType(Span),
    InvalidRayQueryPointer(Span),
    Pointer(&'static str, Span),
    NotPointer(Span),
    NotReference(&'static str, Span),
    InvalidAssignment {
        span: Span,
        ty: InvalidAssignmentType,
    },
    ReservedKeyword(Span),
    /// Redefinition of an identifier (used for both module-scope and local redefinitions).
    Redefinition {
        /// Span of the identifier in the previous definition.
        previous: Span,

        /// Span of the identifier in the new definition.
        current: Span,
    },
    /// A declaration refers to itself directly.
    RecursiveDeclaration {
        /// The location of the name of the declaration.
        ident: Span,

        /// The point at which it is used.
        usage: Span,
    },
    /// A declaration refers to itself indirectly, through one or more other
    /// definitions.
    CyclicDeclaration {
        /// The location of the name of some declaration in the cycle.
        ident: Span,

        /// The edges of the cycle of references.
        ///
        /// Each `(decl, reference)` pair indicates that the declaration whose
        /// name is `decl` has an identifier at `reference` whose definition is
        /// the next declaration in the cycle. The last pair's `reference` is
        /// the same identifier as `ident`, above.
        path: Vec<(Span, Span)>,
    },
    InvalidSwitchValue {
        uint: bool,
        span: Span,
    },
    CalledEntryPoint(Span),
    WrongArgumentCount {
        span: Span,
        expected: Range<u32>,
        found: u32,
    },
    FunctionReturnsVoid(Span),
    InvalidWorkGroupUniformLoad(Span),
    Other,
    ExpectedArraySize(Span),
    NonPositiveArrayLength(Span),
}

impl<'a> Error<'a> {
    pub(crate) fn as_parse_error(&self, source: &'a str) -> ParseError {
        match *self {
            Error::Unexpected(unexpected_span, expected) => {
                let expected_str = match expected {
                    ExpectedToken::Token(token) => {
                        match token {
                            Token::Separator(c) => format!("'{c}'"),
                            Token::Paren(c) => format!("'{c}'"),
                            Token::Attribute => "@".to_string(),
                            Token::Number(_) => "number".to_string(),
                            Token::Word(s) => s.to_string(),
                            Token::Operation(c) => format!("operation ('{c}')"),
                            Token::LogicalOperation(c) => format!("logical operation ('{c}')"),
                            Token::ShiftOperation(c) => format!("bitshift ('{c}{c}')"),
                            Token::AssignmentOperation(c) if c=='<' || c=='>' => format!("bitshift ('{c}{c}=')"),
                            Token::AssignmentOperation(c) => format!("operation ('{c}=')"),
                            Token::IncrementOperation => "increment operation".to_string(),
                            Token::DecrementOperation => "decrement operation".to_string(),
                            Token::Arrow => "->".to_string(),
                            Token::Unknown(c) => format!("unknown ('{c}')"),
                            Token::Trivia => "trivia".to_string(),
                            Token::End => "end".to_string(),
                        }
                    }
                    ExpectedToken::Identifier => "identifier".to_string(),
                    ExpectedToken::Number => "32-bit signed integer literal".to_string(),
                    ExpectedToken::Integer => "unsigned/signed integer literal".to_string(),
                    ExpectedToken::PrimaryExpression => "expression".to_string(),
                    ExpectedToken::Assignment => "assignment or increment/decrement".to_string(),
                    ExpectedToken::SwitchItem => "switch item ('case' or 'default') or a closing curly bracket to signify the end of the switch statement ('}')".to_string(),
                    ExpectedToken::WorkgroupSizeSeparator => "workgroup size separator (',') or a closing parenthesis".to_string(),
                    ExpectedToken::GlobalItem => "global item ('struct', 'const', 'var', 'alias', ';', 'fn') or the end of the file".to_string(),
                    ExpectedToken::Type => "type".to_string(),
                    ExpectedToken::Variable => "variable access".to_string(),
                    ExpectedToken::Function => "function name".to_string(),
                };
                ParseError {
                    message: format!(
                        "expected {}, found '{}'",
                        expected_str, &source[unexpected_span],
                    ),
                    labels: vec![(unexpected_span, format!("expected {expected_str}").into())],
                    notes: vec![],
                }
            }
            Error::UnexpectedComponents(bad_span) => ParseError {
                message: "unexpected components".to_string(),
                labels: vec![(bad_span, "unexpected components".into())],
                notes: vec![],
            },
            Error::UnexpectedOperationInConstContext(span) => ParseError {
                message: "this operation is not supported in a const context".to_string(),
                labels: vec![(span, "operation not supported here".into())],
                notes: vec![],
            },
            Error::BadNumber(bad_span, ref err) => ParseError {
                message: format!("{}: `{}`", err, &source[bad_span],),
                labels: vec![(bad_span, err.to_string().into())],
                notes: vec![],
            },
            Error::NegativeInt(bad_span) => ParseError {
                message: format!(
                    "expected non-negative integer literal, found `{}`",
                    &source[bad_span],
                ),
                labels: vec![(bad_span, "expected non-negative integer".into())],
                notes: vec![],
            },
            Error::BadU32Constant(bad_span) => ParseError {
                message: format!(
                    "expected unsigned integer constant expression, found `{}`",
                    &source[bad_span],
                ),
                labels: vec![(bad_span, "expected unsigned integer".into())],
                notes: vec![],
            },
            Error::BadMatrixScalarKind(span, kind, width) => ParseError {
                message: format!(
                    "matrix scalar type must be floating-point, but found `{}`",
                    kind.to_wgsl(width)
                ),
                labels: vec![(span, "must be floating-point (e.g. `f32`)".into())],
                notes: vec![],
            },
            Error::BadAccessor(accessor_span) => ParseError {
                message: format!("invalid field accessor `{}`", &source[accessor_span],),
                labels: vec![(accessor_span, "invalid accessor".into())],
                notes: vec![],
            },
            Error::UnknownIdent(ident_span, ident) => ParseError {
                message: format!("no definition in scope for identifier: '{ident}'"),
                labels: vec![(ident_span, "unknown identifier".into())],
                notes: vec![],
            },
            Error::UnknownScalarType(bad_span) => ParseError {
                message: format!("unknown scalar type: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown scalar type".into())],
                notes: vec!["Valid scalar types are f32, f64, i32, u32, bool".into()],
            },
            Error::BadTextureSampleType { span, kind, width } => ParseError {
                message: format!(
                    "texture sample type must be one of f32, i32 or u32, but found {}",
                    kind.to_wgsl(width)
                ),
                labels: vec![(span, "must be one of f32, i32 or u32".into())],
                notes: vec![],
            },
            Error::BadIncrDecrReferenceType(span) => ParseError {
                message:
                    "increment/decrement operation requires reference type to be one of i32 or u32"
                        .to_string(),
                labels: vec![(span, "must be a reference type of i32 or u32".into())],
                notes: vec![],
            },
            Error::BadTexture(bad_span) => ParseError {
                message: format!(
                    "expected an image, but found '{}' which is not an image",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "not an image".into())],
                notes: vec![],
            },
            Error::BadTypeCast {
                span,
                ref from_type,
                ref to_type,
            } => {
                let msg = format!("cannot cast a {from_type} to a {to_type}");
                ParseError {
                    message: msg.clone(),
                    labels: vec![(span, msg.into())],
                    notes: vec![],
                }
            }
            Error::InvalidResolve(ref resolve_error) => ParseError {
                message: resolve_error.to_string(),
                labels: vec![],
                notes: vec![],
            },
            Error::InvalidForInitializer(bad_span) => ParseError {
                message: format!(
                    "for(;;) initializer is not an assignment or a function call: '{}'",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "not an assignment or function call".into())],
                notes: vec![],
            },
            Error::InvalidBreakIf(bad_span) => ParseError {
                message: "A break if is only allowed in a continuing block".to_string(),
                labels: vec![(bad_span, "not in a continuing block".into())],
                notes: vec![],
            },
            Error::InvalidGatherComponent(bad_span) => ParseError {
                message: format!(
                    "textureGather component '{}' doesn't exist, must be 0, 1, 2, or 3",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "invalid component".into())],
                notes: vec![],
            },
            Error::InvalidConstructorComponentType(bad_span, component) => ParseError {
                message: format!("invalid type for constructor component at index [{component}]"),
                labels: vec![(bad_span, "invalid component type".into())],
                notes: vec![],
            },
            Error::InvalidIdentifierUnderscore(bad_span) => ParseError {
                message: "Identifier can't be '_'".to_string(),
                labels: vec![(bad_span, "invalid identifier".into())],
                notes: vec![
                    "Use phony assignment instead ('_ =' notice the absence of 'let' or 'var')"
                        .to_string(),
                ],
            },
            Error::ReservedIdentifierPrefix(bad_span) => ParseError {
                message: format!(
                    "Identifier starts with a reserved prefix: '{}'",
                    &source[bad_span]
                ),
                labels: vec![(bad_span, "invalid identifier".into())],
                notes: vec![],
            },
            Error::UnknownAddressSpace(bad_span) => ParseError {
                message: format!("unknown address space: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown address space".into())],
                notes: vec![],
            },
            Error::UnknownAttribute(bad_span) => ParseError {
                message: format!("unknown attribute: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown attribute".into())],
                notes: vec![],
            },
            Error::UnknownBuiltin(bad_span) => ParseError {
                message: format!("unknown builtin: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown builtin".into())],
                notes: vec![],
            },
            Error::UnknownAccess(bad_span) => ParseError {
                message: format!("unknown access: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown access".into())],
                notes: vec![],
            },
            Error::UnknownStorageFormat(bad_span) => ParseError {
                message: format!("unknown storage format: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown storage format".into())],
                notes: vec![],
            },
            Error::UnknownConservativeDepth(bad_span) => ParseError {
                message: format!("unknown conservative depth: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown conservative depth".into())],
                notes: vec![],
            },
            Error::UnknownType(bad_span) => ParseError {
                message: format!("unknown type: '{}'", &source[bad_span]),
                labels: vec![(bad_span, "unknown type".into())],
                notes: vec![],
            },
            Error::SizeAttributeTooLow(bad_span, min_size) => ParseError {
                message: format!("struct member size must be at least {min_size}"),
                labels: vec![(bad_span, format!("must be at least {min_size}").into())],
                notes: vec![],
            },
            Error::AlignAttributeTooLow(bad_span, min_align) => ParseError {
                message: format!("struct member alignment must be at least {min_align}"),
                labels: vec![(bad_span, format!("must be at least {min_align}").into())],
                notes: vec![],
            },
            Error::NonPowerOfTwoAlignAttribute(bad_span) => ParseError {
                message: "struct member alignment must be a power of 2".to_string(),
                labels: vec![(bad_span, "must be a power of 2".into())],
                notes: vec![],
            },
            Error::InconsistentBinding(span) => ParseError {
                message: "input/output binding is not consistent".to_string(),
                labels: vec![(span, "input/output binding is not consistent".into())],
                notes: vec![],
            },
            Error::TypeNotConstructible(span) => ParseError {
                message: format!("type `{}` is not constructible", &source[span]),
                labels: vec![(span, "type is not constructible".into())],
                notes: vec![],
            },
            Error::TypeNotInferrable(span) => ParseError {
                message: "type can't be inferred".to_string(),
                labels: vec![(span, "type can't be inferred".into())],
                notes: vec![],
            },
            Error::InitializationTypeMismatch(name_span, ref expected_ty, ref got_ty) => {
                ParseError {
                    message: format!(
                        "the type of `{}` is expected to be `{}`, but got `{}`",
                        &source[name_span], expected_ty, got_ty,
                    ),
                    labels: vec![(
                        name_span,
                        format!("definition of `{}`", &source[name_span]).into(),
                    )],
                    notes: vec![],
                }
            }
            Error::MissingType(name_span) => ParseError {
                message: format!("variable `{}` needs a type", &source[name_span]),
                labels: vec![(
                    name_span,
                    format!("definition of `{}`", &source[name_span]).into(),
                )],
                notes: vec![],
            },
            Error::MissingAttribute(name, name_span) => ParseError {
                message: format!(
                    "variable `{}` needs a '{}' attribute",
                    &source[name_span], name
                ),
                labels: vec![(
                    name_span,
                    format!("definition of `{}`", &source[name_span]).into(),
                )],
                notes: vec![],
            },
            Error::InvalidAtomicPointer(span) => ParseError {
                message: "atomic operation is done on a pointer to a non-atomic".to_string(),
                labels: vec![(span, "atomic pointer is invalid".into())],
                notes: vec![],
            },
            Error::InvalidAtomicOperandType(span) => ParseError {
                message: "atomic operand type is inconsistent with the operation".to_string(),
                labels: vec![(span, "atomic operand type is invalid".into())],
                notes: vec![],
            },
            Error::InvalidRayQueryPointer(span) => ParseError {
                message: "ray query operation is done on a pointer to a non-ray-query".to_string(),
                labels: vec![(span, "ray query pointer is invalid".into())],
                notes: vec![],
            },
            Error::NotPointer(span) => ParseError {
                message: "the operand of the `*` operator must be a pointer".to_string(),
                labels: vec![(span, "expression is not a pointer".into())],
                notes: vec![],
            },
            Error::NotReference(what, span) => ParseError {
                message: format!("{what} must be a reference"),
                labels: vec![(span, "expression is not a reference".into())],
                notes: vec![],
            },
            Error::InvalidAssignment { span, ty } => {
                let (extra_label, notes) = match ty {
                    InvalidAssignmentType::Swizzle => (
                        None,
                        vec![
                            "WGSL does not support assignments to swizzles".into(),
                            "consider assigning each component individually".into(),
                        ],
                    ),
                    InvalidAssignmentType::ImmutableBinding(binding_span) => (
                        Some((binding_span, "this is an immutable binding".into())),
                        vec![format!(
                            "consider declaring '{}' with `var` instead of `let`",
                            &source[binding_span]
                        )],
                    ),
                    InvalidAssignmentType::Other => (None, vec![]),
                };

                ParseError {
                    message: "invalid left-hand side of assignment".into(),
                    labels: std::iter::once((span, "cannot assign to this expression".into()))
                        .chain(extra_label)
                        .collect(),
                    notes,
                }
            }
            Error::Pointer(what, span) => ParseError {
                message: format!("{what} must not be a pointer"),
                labels: vec![(span, "expression is a pointer".into())],
                notes: vec![],
            },
            Error::ReservedKeyword(name_span) => ParseError {
                message: format!("name `{}` is a reserved keyword", &source[name_span]),
                labels: vec![(
                    name_span,
                    format!("definition of `{}`", &source[name_span]).into(),
                )],
                notes: vec![],
            },
            Error::Redefinition { previous, current } => ParseError {
                message: format!("redefinition of `{}`", &source[current]),
                labels: vec![
                    (
                        current,
                        format!("redefinition of `{}`", &source[current]).into(),
                    ),
                    (
                        previous,
                        format!("previous definition of `{}`", &source[previous]).into(),
                    ),
                ],
                notes: vec![],
            },
            Error::RecursiveDeclaration { ident, usage } => ParseError {
                message: format!("declaration of `{}` is recursive", &source[ident]),
                labels: vec![(ident, "".into()), (usage, "uses itself here".into())],
                notes: vec![],
            },
            Error::CyclicDeclaration { ident, ref path } => ParseError {
                message: format!("declaration of `{}` is cyclic", &source[ident]),
                labels: path
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &(ident, usage))| {
                        [
                            (ident, "".into()),
                            (
                                usage,
                                if i == path.len() - 1 {
                                    "ending the cycle".into()
                                } else {
                                    format!("uses `{}`", &source[ident]).into()
                                },
                            ),
                        ]
                    })
                    .collect(),
                notes: vec![],
            },
            Error::InvalidSwitchValue { uint, span } => ParseError {
                message: "invalid switch value".to_string(),
                labels: vec![(
                    span,
                    if uint {
                        "expected unsigned integer"
                    } else {
                        "expected signed integer"
                    }
                    .into(),
                )],
                notes: vec![if uint {
                    format!("suffix the integer with a `u`: '{}u'", &source[span])
                } else {
                    let span = span.to_range().unwrap();
                    format!(
                        "remove the `u` suffix: '{}'",
                        &source[span.start..span.end - 1]
                    )
                }],
            },
            Error::CalledEntryPoint(span) => ParseError {
                message: "entry point cannot be called".to_string(),
                labels: vec![(span, "entry point cannot be called".into())],
                notes: vec![],
            },
            Error::WrongArgumentCount {
                span,
                ref expected,
                found,
            } => ParseError {
                message: format!(
                    "wrong number of arguments: expected {}, found {}",
                    if expected.len() < 2 {
                        format!("{}", expected.start)
                    } else {
                        format!("{}..{}", expected.start, expected.end)
                    },
                    found
                ),
                labels: vec![(span, "wrong number of arguments".into())],
                notes: vec![],
            },
            Error::FunctionReturnsVoid(span) => ParseError {
                message: "function does not return any value".to_string(),
                labels: vec![(span, "".into())],
                notes: vec![
                    "perhaps you meant to call the function in a separate statement?".into(),
                ],
            },
            Error::InvalidWorkGroupUniformLoad(span) => ParseError {
                message: "incorrect type passed to workgroupUniformLoad".into(),
                labels: vec![(span, "".into())],
                notes: vec!["passed type must be a workgroup pointer".into()],
            },
            Error::Other => ParseError {
                message: "other error".to_string(),
                labels: vec![],
                notes: vec![],
            },
            Error::ExpectedArraySize(span) => ParseError {
                message: "array element count must resolve to an integer scalar (u32 or i32)"
                    .to_string(),
                labels: vec![(span, "must resolve to u32/i32".into())],
                notes: vec![],
            },
            Error::NonPositiveArrayLength(span) => ParseError {
                message: "array element count must be greater than zero".to_string(),
                labels: vec![(span, "must be greater than zero".into())],
                notes: vec![],
            },
        }
    }
}
