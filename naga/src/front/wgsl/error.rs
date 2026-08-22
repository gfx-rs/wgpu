//! Formatting WGSL front end error messages.

#![expect(
    clippy::pattern_type_mismatch,
    reason = "There are matches on references, since it produces less LLVM IR than dereferencing"
)]

use crate::common::wgsl::TryToWgsl;
use crate::diagnostic_filter::ConflictingDiagnosticRuleError;
use crate::error::replace_control_chars;
use crate::proc::{Alignment, ConstantEvaluatorError, ResolveError};
use crate::{Scalar, SourceLocation, Span, UnaryOperator};

use super::parse::directive::enable_extension::{EnableExtension, UnimplementedEnableExtension};
use super::parse::directive::language_extension::{
    LanguageExtension, UnimplementedLanguageExtension,
};
use super::parse::lexer::Token;

use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use thiserror::Error;

use alloc::{
    borrow::Cow,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Write as _;
use core::ops::Range;

#[derive(Clone, Debug)]
pub struct ParseError {
    message: Cow<'static, str>,
    // The first span should be the primary span, and the other ones should be complementary.
    labels: Vec<(Span, Cow<'static, str>)>,
    notes: Vec<Cow<'static, str>>,
}

impl ParseError {
    pub fn labels(&self) -> impl ExactSizeIterator<Item = (Span, &str)> + '_ {
        self.labels
            .iter()
            .map(|&(span, ref msg)| (span, msg.as_ref()))
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn notes(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.notes.iter().map(Cow::as_ref)
    }

    fn diagnostic(&self) -> Diagnostic<()> {
        let diagnostic = Diagnostic::error()
            .with_message(self.message.to_string())
            .with_labels(
                self.labels
                    .iter()
                    .filter_map(|label| label.0.to_range().map(|range| (label, range)))
                    .map(|(label, range)| {
                        Label::primary((), range).with_message(label.1.to_string())
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
    #[cfg(feature = "stderr")]
    pub fn emit_to_stderr(&self, source: &str) {
        self.emit_to_stderr_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to standard error stream.
    #[cfg(feature = "stderr")]
    pub fn emit_to_stderr_with_path<P>(&self, source: &str, path: P)
    where
        P: AsRef<std::path::Path>,
    {
        let path = path.as_ref().display().to_string();
        let files = SimpleFile::new(path, replace_control_chars(source));
        let config = term::Config::default();

        cfg_if::cfg_if! {
            if #[cfg(feature = "termcolor")] {
                let writer = term::termcolor::StandardStream::stderr(term::termcolor::ColorChoice::Auto);
                term::emit_to_write_style(&mut writer.lock(), &config, &files, &self.diagnostic())
                    .expect("cannot write error");
            } else {
                let writer = std::io::stderr();
                term::emit_to_io_write(&mut writer.lock(), &config, &files, &self.diagnostic())
                    .expect("cannot write error");
            }
        }
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string(&self, source: &str) -> String {
        self.emit_to_string_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to a string.
    ///
    /// `path` gives the filename to attribute the error to in the
    /// output; this function does not try to access the file.
    pub fn emit_to_string_with_path(&self, source: &str, path: &str) -> String {
        let files = SimpleFile::new(path, replace_control_chars(source));
        let config = term::Config::default();

        let mut writer = crate::error::DiagnosticBuffer::new();
        writer
            .emit_to_self(&config, &files, &self.diagnostic())
            .expect("cannot write error");
        writer.into_string()
    }

    /// Returns a [`SourceLocation`] for the first label in the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        self.labels.first().map(|label| label.0.location(source))
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl core::error::Error for ParseError {}

#[cfg(test)]
mod parse_error_tests {

    #[test]
    fn test_notes() {
        use crate::front::wgsl::parse_str;
        // wgsl code and notes taken from: `cross_vec2()` in naga/tests/naga/wgsl_errors.rs
        assert_eq!(
            parse_str(
                r#"
            fn x() -> f32 {
                return cross(vec2(0., 1.), vec2(0., 1.));
            }
        "#,
            )
            .unwrap_err()
            .notes()
            .collect::<super::Vec<_>>(),
            [
                "`cross` accepts the following types for argument #1:",
                "allowed type: vec3<{AbstractFloat}>",
                "allowed type: vec3<f32>",
                "allowed type: vec3<f16>",
                "allowed type: vec3<f64>"
            ]
            .to_vec()
        );
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    AfterIdentListComma,
    AfterIdentListArg,
    /// LHS expression (identifier component_or_swizzle_specifier?, (`lhs_expression`) component_or_swizzle_specifier?, &`lhs_expression`, *`lhs_expression`)
    LhsExpression,
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
    /// Access of `var`, `let`, `const`.
    Variable,
    /// Access of a function
    Function,
    /// The `diagnostic` identifier of the `@diagnostic(…)` attribute.
    DiagnosticAttribute,
    /// statement
    Statement,
    /// for loop init statement (variable_or_value_statement, variable_updating_statement, func_call_statement)
    ForInit,
    /// for loop update statement (variable_updating_statement, func_call_statement)
    ForUpdate,
}

impl core::fmt::Display for ExpectedToken<'_> {
    #[expect(
        unused,
        reason = "This ignores write errors, since this should only be called to write into a \
                  String, which is infallible. Ignoring errors lowers binary bloat from this \
                  function."
    )]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        // This function uses a temporary enum here to lower binary bloat by merging calls to
        // write_str.
        //
        // Tokens where the output is already a single string (such as Token::Number) use Kind::Str.
        // Tokens where the output is made up of two static strings split by a char (such as
        // Token::Operation) use Kind::FormatChar
        // Tokens where the output is 3 strings concatenated (such as Token::DocComment) use
        // Kind::FormatStr
        //
        // Any exceptions (such as Token::ShiftOperation) handle their cases explicitly and return
        enum Kind<'a> {
            Str(&'a str),
            FormatChar(&'static str, char, &'static str),
            FormatStr([&'a str; 3]),
        }

        let kind = match *self {
            ExpectedToken::Token(token) => match token {
                Token::Separator(c) | Token::Paren(c) => Kind::FormatChar("`", c, "`"),
                Token::Attribute => Kind::Str("@"),
                Token::Number(_) => Kind::Str("number"),
                Token::Word(s) => Kind::Str(s),
                Token::Operation(c) => Kind::FormatChar("operation (`", c, "`)"),
                Token::LogicalOperation(c) => Kind::FormatChar("logical operation (`", c, "`)"),
                Token::ShiftOperation(c) => {
                    f.write_str("bitshift (`");
                    f.write_char(c);
                    f.write_char(c);
                    f.write_str("`)");
                    return Ok(())
                },
                Token::AssignmentOperation(c) if c == '<' || c == '>' => {
                    f.write_str("bitshift (`");
                    f.write_char(c);
                    f.write_char(c);
                    f.write_str("=`)");
                    return Ok(())
                }
                Token::AssignmentOperation(c) => Kind::FormatChar("operation (`", c, "=`)"),
                Token::IncrementOperation => Kind::Str("increment operation"),
                Token::DecrementOperation => Kind::Str("decrement operation"),
                Token::Arrow => Kind::Str("->"),
                Token::TemplateArgsStart => Kind::Str("template args start"),
                Token::TemplateArgsEnd => Kind::Str("template args end"),
                Token::Unknown(c) => Kind::FormatChar("unknown (`", c, "`)"),
                Token::Trivia => Kind::Str("trivia"),
                Token::DocComment(s) => Kind::FormatStr(["doc comment ('", s, "')"]),
                Token::ModuleDocComment(s) => Kind::FormatStr(["module doc comment ('", s, "')"]),
                Token::End => Kind::Str("end"),
                Token::UnterminatedBlockComment(s) => {
                    Kind::FormatStr(["unterminated doc comment ('", s, "')"])
                }
            },
            ExpectedToken::Identifier => Kind::Str("identifier"),
            ExpectedToken::LhsExpression => Kind::Str("LHS expression (identifier component_or_swizzle_specifier?, (`lhs_expression`) component_or_swizzle_specifier?, &`lhs_expression`, *`lhs_expression`)"),
            ExpectedToken::PrimaryExpression => Kind::Str("expression"),
            ExpectedToken::Assignment => Kind::Str("assignment or increment/decrement"),
            ExpectedToken::SwitchItem => Kind::Str(concat!(
                "switch item (`case` or `default`) or a closing curly bracket ",
                "to signify the end of the switch statement (`}`)"
            )),
            ExpectedToken::WorkgroupSizeSeparator => {
                Kind::Str("workgroup size separator (`,`) or a closing parenthesis")
            }
            ExpectedToken::GlobalItem => Kind::Str(concat!(
                "global item (`struct`, `const`, `var`, `alias`, ",
                "`fn`, `diagnostic`, `enable`, `requires`, `;`) ",
                "or the end of the file"
            )),
            ExpectedToken::Variable => Kind::Str("variable access"),
            ExpectedToken::Function => Kind::Str("function name"),
            ExpectedToken::AfterIdentListArg => {
                Kind::Str("next argument, trailing comma, or end of list (`,` or `;`)")
            }
            ExpectedToken::AfterIdentListComma => {
                Kind::Str("next argument or end of list (`;`)")
            }
            ExpectedToken::DiagnosticAttribute => {
                Kind::Str("the `diagnostic` attribute identifier")
            }
            ExpectedToken::Statement => Kind::Str("statement"),
            ExpectedToken::ForInit => Kind::Str("for loop initializer statement (`var`/`let`/`const` declaration, assignment, `i++`/`i--` statement, function call)"),
            ExpectedToken::ForUpdate => Kind::Str("for loop update statement (assignment, `i++`/`i--` statement, function call)"),
        };

        match kind {
            Kind::Str(s) => {
                f.write_str(s);
            }
            Kind::FormatChar(a, b, c) => {
                f.write_str(a);
                f.write_char(b);
                f.write_str(c);
            }
            Kind::FormatStr(strings) => {
                for s in strings {
                    f.write_str(s);
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NumberError {
    #[error("invalid numeric literal format")]
    Invalid,
    #[error("numeric literal not representable by target type")]
    NotRepresentable,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InvalidAssignmentType {
    Other,
    Swizzle,
    ImmutableBinding(Span),
}

#[derive(Clone, Debug)]
pub(crate) enum Error<'a> {
    Unexpected(Span, ExpectedToken<'a>),
    UnexpectedComponents(Span),
    UnexpectedOperationInConstContext(Span),
    BadNumber(Span, NumberError),
    BadMatrixScalarKind(Span, Scalar),
    BadAccessor(Span),
    BadTexture(Span),
    BadTypeCast {
        span: Span,
        from_type: String,
        to_type: String,
    },
    NotStorageTexture(Span),
    BadTextureSampleType {
        span: Span,
        scalar: Scalar,
    },
    BadIncrDecrReferenceType(Span),
    InvalidResolve(ResolveError),
    /// A break if appeared outside of a continuing block
    InvalidBreakIf(Span),
    InvalidGatherComponent(Span),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    InvalidLocalVariableAddressSpace(Span),
    UnknownRayFlag(Span),
    RepeatedAttribute(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    UnknownEnableExtension(Span, &'a str),
    UnknownLanguageExtension(Span, &'a str),
    UnknownDiagnosticRuleName(Span),
    SizeAttributeTooLow(Span, u32),
    SizeAttributeRequiresFixedFootprint(Span),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    TypeNotConstructible(Span),
    TypeNotInferable(Span),
    InitializationTypeMismatch {
        name: Span,
        expected: String,
        got: String,
    },
    DeclMissingTypeAndInit(Span),
    MissingAttribute(&'static str, Span),
    InvalidUnaryOperandType {
        span: Span,
        op: UnaryOperator,
        operand_type: String,
    },

    InvalidAddrOfOperand(Span),
    InvalidAtomicPointer(Span),
    InvalidAtomicOperandType(Span),
    InvalidAtomicAccess(Span),
    InvalidRayQueryPointer(Span),
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
        path: Box<[(Span, Span)]>,
    },
    InvalidSwitchSelector {
        span: Span,
    },
    InvalidSwitchCase {
        span: Span,
    },
    SwitchCaseTypeMismatch {
        span: Span,
    },
    CalledEntryPoint(Span),
    CalledLocalDecl(Span),
    WrongArgumentCount {
        span: Span,
        expected: Range<u32>,
        found: u32,
    },
    /// No overload of this function accepts this many arguments.
    TooManyArguments {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first argument that is unacceptable.
        arg_span: Span,

        /// Maximum number of arguments accepted by any overload of
        /// this function.
        max_arguments: u32,
    },
    /// A value passed to a builtin function has a type that is not
    /// accepted by any overload of the function.
    WrongArgumentType {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first argument whose type is unacceptable.
        arg_span: Span,

        /// The index of the first argument whose type is unacceptable.
        arg_index: u32,

        /// That argument's actual type.
        arg_ty: String,

        /// The set of argument types that would have been accepted for
        /// this argument, given the prior arguments.
        allowed: Vec<String>,
    },
    /// A value passed to a builtin function has a type that is not
    /// accepted, given the earlier arguments' types.
    InconsistentArgumentType {
        /// The name of the function being called.
        function: String,

        /// The function name in the call expression.
        call_span: Span,

        /// The first unacceptable argument.
        arg_span: Span,

        /// The index of the first unacceptable argument.
        arg_index: u32,

        /// The actual type of the first unacceptable argument.
        arg_ty: String,

        /// The prior argument whose type made the `arg_span` argument
        /// unacceptable.
        inconsistent_span: Span,

        /// The index of the `inconsistent_span` argument.
        inconsistent_index: u32,

        /// The type of the `inconsistent_span` argument.
        inconsistent_ty: String,

        /// The types that would have been accepted instead of the
        /// first unacceptable argument.
        allowed: Vec<String>,
    },
    FunctionReturnsVoid(Span),
    FunctionMustUseUnused(Span),
    FunctionMustUseReturnsVoid(Span, Span),
    FunctionMustUseOnNonFunction(Span),
    InvalidWorkGroupUniformLoad(Span),
    Internal(&'static str),
    ExpectedConstExprConcreteIntegerScalar(Span),
    ExpectedNonNegative(Span),
    ExpectedPositiveArrayLength(Span),
    MissingWorkgroupSize(Span),
    ConstantEvaluatorError(Box<ConstantEvaluatorError>, Span),
    AutoConversion(Box<AutoConversionError>),
    AutoConversionLeafScalar(Box<AutoConversionLeafScalarError>),
    ConcretizationFailed(Box<ConcretizationFailedError>),
    ExceededLimitForNestedBraces {
        span: Span,
        limit: u8,
    },
    PipelineConstantIDValue(Span),
    NotBool(Span),
    ConstAssertFailed(Span),
    DirectiveAfterFirstGlobalDecl {
        directive_span: Span,
    },
    EnableExtensionNotYetImplemented {
        kind: UnimplementedEnableExtension,
        span: Span,
    },
    EnableExtensionNotEnabled {
        kind: EnableExtension,
        span: Span,
    },
    EnableExtensionNotSupported {
        kind: EnableExtension,
        span: Span,
    },
    LanguageExtensionNotYetImplemented {
        kind: UnimplementedLanguageExtension,
        span: Span,
    },
    DiagnosticInvalidSeverity {
        severity_control_name_span: Span,
    },
    DiagnosticDuplicateTriggeringRule(ConflictingDiagnosticRuleError),
    DiagnosticAttributeNotYetImplementedAtParseSite {
        site_name_plural: &'static str,
        spans: Vec<Span>,
    },
    DiagnosticAttributeNotSupported {
        on_what: DiagnosticAttributeNotSupportedPosition,
        spans: Vec<Span>,
    },
    SelectUnexpectedArgumentType {
        arg_span: Span,
        arg_type: String,
    },
    SelectRejectAndAcceptHaveNoCommonType {
        reject_span: Span,
        reject_type: String,
        accept_span: Span,
        accept_type: String,
    },
    ExpectedGlobalVariable {
        name_span: Span,
    },
    StructMemberTooLarge {
        member_name_span: Span,
    },
    TypeTooLarge {
        span: Span,
    },
    UnderspecifiedCooperativeMatrix,
    InvalidCooperativeLoadType(Span),
    UnsupportedCooperativeScalar(Span),
    UnexpectedIdentForEnumerant(Span),
    UnexpectedExprForEnumerant(Span),
    UnusedArgsForTemplate(Vec<Span>),
    UnexpectedTemplate(Span),
    MissingTemplateArg {
        span: Span,
        description: &'static str,
    },
    UnexpectedExprForTypeExpression(Span),
    MissingIncomingPayload(Span),
    UnterminatedBlockComment(Span),
}

impl From<ConflictingDiagnosticRuleError> for Error<'_> {
    fn from(value: ConflictingDiagnosticRuleError) -> Self {
        Self::DiagnosticDuplicateTriggeringRule(value)
    }
}

/// Used for diagnostic refinement in [`Error::DiagnosticAttributeNotSupported`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum DiagnosticAttributeNotSupportedPosition {
    SemicolonInModulePosition,
    Other { display_plural: &'static str },
}

impl From<&'static str> for DiagnosticAttributeNotSupportedPosition {
    fn from(display_plural: &'static str) -> Self {
        Self::Other { display_plural }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AutoConversionError {
    pub dest_span: Span,
    pub dest_type: String,
    pub source_span: Span,
    pub source_type: String,
}

#[derive(Clone, Debug)]
pub(crate) struct AutoConversionLeafScalarError {
    pub dest_span: Span,
    pub dest_scalar: String,
    pub source_span: Span,
    pub source_type: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ConcretizationFailedError {
    pub expr_span: Span,
    pub expr_type: String,
    pub concretization_preferences: Vec<(String, ConstantEvaluatorError)>,
}

impl<'a> Error<'a> {
    #[cold]
    #[inline(never)]
    // This function is not recursive, so it can exceed our conservatively-set
    // `stack-frame-limit` without causing problems.
    #[allow(clippy::large_stack_frames)]
    pub(crate) fn as_parse_error(&self, source: &'a str) -> ParseError {
        // This function normally produces a lot of binary bloat, so has been structured in a
        // non-idiomatic way to minimize its size.
        //
        // Errors where the output has a static message and a single, static label are handled in
        // the block starting with UnexpectedComponents.
        //
        // Errors where the output has a single static label, and a message in the form
        // "..{&source[span]}.." are handled in the block starting with BadAccessor.
        //
        // Other errors with a single static label, and a non-static message are handled in the
        // block starting with BadMatrixScalarKind.
        //
        // Other errors are handled on their own.
        //
        // If you're adding a new error type, and you're not sure where to put it, you can put it
        // on its own; there'll only be a small difference in size from one variant :)

        match self {
            Error::Unexpected(unexpected_span, expected) => {
                ParseError {
                    message: format!(
                        "expected {expected}, found {:?}",
                        &source[*unexpected_span]
                    ).into(),
                    labels: vec![(*unexpected_span, format!("expected {expected}").into())],
                    notes: vec![],
                }
            }
            // Variants where message and label are static strings
            Error::UnexpectedComponents(span)
                | Error::UnexpectedOperationInConstContext(span)
                | Error::NotStorageTexture(span)
                | Error::BadIncrDecrReferenceType(span)
                | Error::InvalidBreakIf(span)
                | Error::NonPowerOfTwoAlignAttribute(span)
                | Error::InconsistentBinding(span)
                | Error::TypeNotInferable(span)
                | Error::InvalidAddrOfOperand(span)
                | Error::InvalidAtomicPointer(span)
                | Error::InvalidAtomicOperandType(span)
                | Error::InvalidAtomicAccess(span)
                | Error::InvalidRayQueryPointer(span)
                | Error::NotPointer(span)
                | Error::InvalidSwitchSelector { span }
                | Error::InvalidSwitchCase { span }
                | Error::SwitchCaseTypeMismatch { span }
                | Error::CalledEntryPoint(span)
                | Error::CalledLocalDecl(span)
                | Error::ExpectedConstExprConcreteIntegerScalar(span)
                | Error::ExpectedNonNegative(span)
                | Error::ExpectedPositiveArrayLength(span)
                | Error::MissingWorkgroupSize(span)
                | Error::PipelineConstantIDValue(span)
                | Error::NotBool(span)
                | Error::ConstAssertFailed(span)
                | Error::ExpectedGlobalVariable { name_span: span }
                | Error::UnexpectedExprForEnumerant(span)
                | Error::UnexpectedTemplate(span)
                | Error::UnexpectedExprForTypeExpression(span)
                | Error::MissingIncomingPayload(span) => {
                let (message, label) = match self {
                    Error::UnexpectedComponents(_) => (
                        "unexpected components",
                        "unexpected components"
                    ),
                    Error::UnexpectedOperationInConstContext(_) => (
                        "this operation is not supported in a const context",
                        "operation not supported here"
                    ),
                    Error::NotStorageTexture(_) => (
                        "textureStore can only be applied to storage textures",
                        "not a storage texture"
                    ),
                    Error::BadIncrDecrReferenceType(_) => (
                        concat!(
                            "increment/decrement operation requires ",
                            "reference type to be one of i32 or u32"
                        ),
                        "must be a reference type of i32 or u32"
                    ),
                    Error::InvalidBreakIf(_) => (
                        "A break if is only allowed in a continuing block",
                        "not in a continuing block"
                    ),
                    Error::NonPowerOfTwoAlignAttribute(_) => (
                        "struct member alignment must be a power of 2",
                        "must be a power of 2"
                    ),
                    Error::InconsistentBinding(_) => (
                        "input/output binding is not consistent",
                        "input/output binding is not consistent"
                    ),
                    Error::TypeNotInferable(_) => (
                        "type can't be inferred",
                        "type can't be inferred"
                    ),
                    Error::InvalidAddrOfOperand(_) => (
                        "cannot take the address of a vector component",
                        "invalid operand for address-of"
                    ),
                    Error::InvalidAtomicPointer(_) => (
                        "atomic operation is done on a pointer to a non-atomic",
                        "atomic pointer is invalid"
                    ),
                    Error::InvalidAtomicOperandType(_) => (
                        "atomic operand type is inconsistent with the operation",
                        "atomic operand type is invalid"
                    ),
                    Error::InvalidAtomicAccess(_) => (
                        "direct access to atomic variable is not allowed",
                        "atomic variables cannot be accessed directly; use atomic built-in functions",
                    ),
                    Error::InvalidRayQueryPointer(_) => (
                        "ray query operation is done on a pointer to a non-ray-query",
                        "ray query pointer is invalid"
                    ),
                    Error::NotPointer(_) => (
                        "the operand of the `*` operator must be a pointer",
                        "expression is not a pointer"
                    ),
                    Error::InvalidSwitchSelector { .. } => (
                        "invalid `switch` selector",
                        "`switch` selector must be a scalar integer"
                    ),
                    Error::InvalidSwitchCase { .. } => (
                        "invalid `switch` case selector value",
                        "`switch` case selector must be a scalar integer const expression"
                    ),
                    Error::SwitchCaseTypeMismatch { .. } => (
                        "invalid `switch` case selector value",
                        "`switch` case selector must have the same type as the `switch` selector expression"
                    ),
                    Error::CalledEntryPoint(_) => (
                        "entry point cannot be called",
                        "entry point cannot be called"
                    ),
                    Error::CalledLocalDecl(_) => (
                        "local declaration cannot be called",
                        "local declaration cannot be called"
                    ),
                    Error::ExpectedConstExprConcreteIntegerScalar(_) => (
                        concat!(
                            "must be a const-expression that ",
                            "resolves to a concrete integer scalar (`u32` or `i32`)"
                        ),
                        "must resolve to `u32` or `i32`"
                    ),
                    Error::ExpectedNonNegative(_) => (
                        "must be non-negative (>= 0)",
                        "must be non-negative"
                    ),
                    Error::ExpectedPositiveArrayLength(_) => (
                        "array element count must be positive (> 0)",
                        "must be positive"
                    ),
                    Error::MissingWorkgroupSize(_) => (
                        "workgroup size is missing on compute shader entry point",
                        "must be paired with a `@workgroup_size` attribute"
                    ),
                    Error::PipelineConstantIDValue(_) => (
                        "pipeline constant ID must be between 0 and 65535 inclusive",
                        "must be between 0 and 65535 inclusive"
                    ),
                    Error::NotBool(_) => (
                        "must be a const-expression that resolves to a `bool`",
                        "must resolve to `bool`"
                    ),
                    Error::ConstAssertFailed(_) => (
                        "`const_assert` failure",
                        "evaluates to `false`"
                    ),
                    Error::ExpectedGlobalVariable { .. } => (
                        "expected global variable",
                        "variable used here"
                    ),
                    Error::UnexpectedExprForEnumerant(_) => (
                        "unexpected expression",
                        "needs to be an identifier resolving to a predeclared enumerant"
                    ),
                    Error::UnexpectedTemplate(_) => (
                        "unexpected template",
                        "expected identifier"
                    ),
                    Error::UnexpectedExprForTypeExpression(_) => (
                        "unexpected expression",
                        "needs to be an identifier resolving to a type declaration (alias or struct) or predeclared type(-generator)"
                    ),
                    Error::MissingIncomingPayload(_) => (
                        "incoming payload is missing on a `closest_hit`, `any_hit` or `miss` shader entry point",
                        "must be paired with a `@incoming_payload` attribute"
                    ),
                    _ => unreachable!()
                };

                ParseError {
                    labels: vec![(*span, label.into())],
                    message: message.into(),
                    notes: vec![],
                }
            },
            // Variants with a static label, and a message in the form "..{&source[span]}.."
            Error::BadAccessor(span)
                | Error::BadTexture(span)
                | Error::InvalidGatherComponent(span)
                | Error::ReservedIdentifierPrefix(span)
                | Error::UnknownAddressSpace(span)
                | Error::InvalidLocalVariableAddressSpace(span)
                | Error::UnknownRayFlag(span)
                | Error::RepeatedAttribute(span)
                | Error::UnknownAttribute(span)
                | Error::UnknownBuiltin(span)
                | Error::UnknownAccess(span)
                | Error::UnknownStorageFormat(span)
                | Error::UnknownConservativeDepth(span)
                | Error::TypeNotConstructible(span)
                | Error::DeclMissingTypeAndInit(span)
                | Error::UnexpectedIdentForEnumerant(span) => {
                let (message_a, message_b, label) = match self {
                    Error::BadAccessor(_) => (
                        "invalid field accessor `",
                        "`",
                        "invalid accessor"
                    ),
                    Error::BadTexture(_) => (
                        "expected an image, but found `",
                        "` which is not an image",
                        "not an image"
                    ),
                    Error::InvalidGatherComponent(_) => (
                        "textureGather component `",
                        "` doesn't exist, must be 0, 1, 2, or 3",
                        "invalid component"
                    ),
                    Error::ReservedIdentifierPrefix(_) => (
                        "Identifier starts with a reserved prefix: `",
                        "`",
                        "invalid identifier"
                    ),
                    Error::UnknownAddressSpace(_) => (
                        "unknown address space: `",
                        "`",
                        "unknown address space"
                    ),
                    Error::InvalidLocalVariableAddressSpace(_) => (
                        "invalid address space for local variable: `",
                        "`",
                        "local variables can only use 'function' address space"
                    ),
                    Error::UnknownRayFlag(_) => (
                        "unknown ray flag: `",
                        "`",
                        "unknown ray flag"
                    ),
                    Error::RepeatedAttribute(_) => (
                        "repeated attribute: `",
                        "`",
                        "repeated attribute"
                    ),
                    Error::UnknownAttribute(_) => (
                        "unknown attribute: `",
                        "`",
                        "unknown attribute"
                    ),
                    Error::UnknownBuiltin(_) => (
                        "unknown builtin: `",
                        "`",
                        "unknown builtin"
                    ),
                    Error::UnknownAccess(_) => (
                        "unknown access: `",
                        "`",
                        "unknown access"
                    ),
                    Error::UnknownStorageFormat(_) => (
                        "unknown storage format: `",
                        "`",
                        "unknown storage format"
                    ),
                    Error::UnknownConservativeDepth(_) => (
                        "unknown conservative depth: `",
                        "`",
                        "unknown conservative depth"
                    ),
                    Error::TypeNotConstructible(_) => (
                        "type `",
                        "` is not constructible",
                        "type is not constructible"
                    ),
                    Error::DeclMissingTypeAndInit(_) => (
                        "declaration of `",
                        "` needs a type specifier or initializer",
                        "needs a type specifier or initializer"
                    ),
                    Error::UnexpectedIdentForEnumerant(_) => (
                        "identifier `",
                        "` resolves to a declaration",
                        "needs to resolve to a predeclared enumerant"
                    ),
                    _ => unreachable!()
                };

                ParseError {
                    message: [message_a, &source[*span], message_b].concat().into(),
                    labels: vec![(*span, label.into())],
                    notes: vec![]
                }
            },
            // Other variants with a single static label and no notes
            Error::BadMatrixScalarKind(span, _)
                | Error::UnknownIdent(span, _)
                | Error::BadTextureSampleType { span, .. }
                | Error::InvalidConstructorComponentType(span, _)
                | Error::NotReference(_, span)
                | Error::WrongArgumentCount { span, .. }
                | Error::ConstantEvaluatorError(_, span)
                | Error::EnableExtensionNotSupported { span, .. }
                | Error::InvalidUnaryOperandType { span, .. }
                | Error::MissingTemplateArg { span, .. } => {
                let (message, label) = match self {
                    Error::BadMatrixScalarKind(_, scalar) => (
                        format!(
                            "matrix scalar type must be floating-point, but found `{}`",
                            scalar.to_wgsl_for_diagnostics()
                        ),
                        "must be floating-point (e.g. `f32`)"
                    ),
                    Error::UnknownIdent(_, ident) => (
                        format!("no definition in scope for identifier: `{ident}`"),
                        "unknown identifier"
                    ),
                    Error::BadTextureSampleType { scalar, .. } => (
                        format!(
                            "texture sample type must be one of f32, i32 or u32, but found {}",
                            scalar.to_wgsl_for_diagnostics()
                        ),
                        "must be one of f32, i32 or u32"
                    ),
                    Error::InvalidConstructorComponentType(_, component) => (
                        format!("invalid type for constructor component at index [{component}]"),
                        "invalid component type"
                    ),
                    Error::NotReference(what, _) => (
                        format!("{what} must be a reference"),
                        "expression is not a reference"
                    ),
                    Error::WrongArgumentCount {
                        ref expected,
                        found,
                        ..
                    } => (
                        format!(
                            "wrong number of arguments: expected {}, found {}",
                            if expected.len() < 2 {
                                format!("{}", expected.start)
                            } else {
                                format!("{}..{}", expected.start, expected.end)
                            },
                            found
                        ),
                        "wrong number of arguments"
                    ),
                    Error::ConstantEvaluatorError(ref e, _) => (e.to_string(), "see msg"),
                    Error::EnableExtensionNotSupported { kind, .. } => (
                        format!(
                            "the `{}` extension is not supported in the current environment",
                            kind.to_ident()
                        ),
                        "unsupported enable-extension"
                    ),
                    Error::MissingTemplateArg { description, .. } => (
                        format!(
                            "`{}` needs a template argument specified: {description}",
                            &source[*span]
                        ),
                        "is missing a template argument"
                    ),
                    Error::InvalidUnaryOperandType { op, operand_type, .. } => {
                        let operator = match op {
                            UnaryOperator::Negate => "-",
                            UnaryOperator::LogicalNot => "!",
                            UnaryOperator::BitwiseNot => "~",
                        };
                        (
                            format!(
                                "unary operator `{operator}` is not defined for operand type `{}`",
                                operand_type
                            ),
                            "invalid operand type for this operator"
                        )
                    },
                    _ => unreachable!()
                };

                ParseError {
                    message: message.into(),
                    labels: vec![(*span, label.into())],
                    notes: vec![]
                }
            },
            Error::BadNumber(bad_span, ref err) => ParseError {
                message: format!("{}: `{}`", err, &source[*bad_span]).into(),
                labels: vec![(*bad_span, err.to_string().into())],
                notes: vec![],
            },
            Error::UnknownScalarType(bad_span) => ParseError {
                message: format!("unknown scalar type: `{}`", &source[*bad_span]).into(),
                labels: vec![(*bad_span, "unknown scalar type".into())],
                notes: vec!["Valid scalar types are f32, f64, i32, u32, bool".into()],
            },
            Error::BadTypeCast {
                span,
                ref from_type,
                ref to_type,
            } => {
                let msg = format!("cannot cast a {from_type} to a {to_type}");
                ParseError {
                    message: msg.clone().into(),
                    labels: vec![(*span, msg.into())],
                    notes: vec![],
                }
            }
            Error::InvalidResolve(ref resolve_error) => ParseError {
                message: resolve_error.to_string().into(),
                labels: vec![],
                notes: vec![],
            },
            Error::InvalidIdentifierUnderscore(bad_span) => ParseError {
                labels: vec![(*bad_span, "invalid identifier".into())],
                notes: vec![
                    "Use phony assignment instead (`_ =` notice the absence of `let` or `var`)"
                        .into(),
                ],
                message: "Identifier can't be `_`".into(),
            },
            Error::UnknownEnableExtension(span, word) => ParseError {
                message: format!("unknown enable-extension `{word}`").into(),
                labels: vec![(*span, "".into())],
                notes: vec![
                    "See available extensions at <https://www.w3.org/TR/WGSL/#enable-extension>."
                        .into(),
                ],
            },
            Error::UnknownLanguageExtension(span, name) => ParseError {
                message: format!("unknown language extension `{name}`").into(),
                labels: vec![(*span, "".into())],
                notes: vec![concat!(
                    "See available extensions at ",
                    "<https://www.w3.org/TR/WGSL/#language-extensions-sec>."
                )
                .into()],
            },
            Error::UnknownDiagnosticRuleName(span) => ParseError {
                message: format!("unknown `diagnostic(…)` rule name `{}`", &source[*span]).into(),
                labels: vec![(*span, "not a valid diagnostic rule name".into())],
                notes: vec![concat!(
                    "See available trigger rules at ",
                    "<https://www.w3.org/TR/WGSL/#filterable-triggering-rules>."
                )
                .into()],
            },
            Error::SizeAttributeTooLow(bad_span, min_size) => ParseError {
                message: format!("struct member size must be at least {min_size}").into(),
                labels: vec![(*bad_span, format!("must be at least {min_size}").into())],
                notes: vec![],
            },
            Error::SizeAttributeRequiresFixedFootprint(bad_span) => ParseError {
                labels: vec![(*bad_span, "type does not have creation-fixed footprint".into())],
                message: "@size attribute requires a type with creation-fixed footprint".into(),
                notes: vec![],
            },
            Error::AlignAttributeTooLow(bad_span, min_align) => ParseError {
                message: format!("struct member alignment must be at least {min_align}").into(),
                labels: vec![(*bad_span, format!("must be at least {min_align}").into())],
                notes: vec![],
            },
            Error::InitializationTypeMismatch {
                name,
                ref expected,
                ref got,
            } => {
                let name_str = &source[*name];
                ParseError {
                    message: format!(
                        "the type of `{name_str}` is expected to be `{expected}`, but got `{got}`"
                    )
                        .into(),
                    labels: vec![(*name, format!("definition of `{name_str}`").into())],
                    notes: vec![],
                }
            },
            Error::MissingAttribute(name, name_span) => {
                let variable = &source[*name_span];
                ParseError {
                    message: format!(
                        "variable `{variable}` needs a '{name}' attribute",
                    )
                        .into(),
                    labels: vec![(
                        *name_span,
                        format!("definition of `{variable}`").into(),
                    )],
                    notes: vec![],
                }
            },
            Error::InvalidAssignment { span, ty } => {
                let (notes, extra_label) = match ty {
                    InvalidAssignmentType::Swizzle => (
                        vec![
                            "WGSL does not support assignments to swizzles".into(),
                            "consider assigning each component individually".into(),
                        ],
                        None,
                    ),
                    InvalidAssignmentType::ImmutableBinding(binding_span) => (
                        vec![format!(
                            "consider declaring `{}` with `var` instead of `let`",
                            &source[*binding_span]
                        ).into()],
                        Some((*binding_span, "this is an immutable binding".into())),
                    ),
                    InvalidAssignmentType::Other => (vec![], None),
                };

                let label = (*span, "cannot assign to this expression".into());

                ParseError {
                    labels: if let Some(extra_label) = extra_label {
                        vec![label, extra_label]
                    } else {
                        vec![label]
                    },
                    message: "invalid left-hand side of assignment".into(),
                    notes,
                }
            }
            Error::ReservedKeyword(name_span) => {
                let name = &source[*name_span];
                ParseError {
                    message: format!("name `{name}` is a reserved keyword").into(),
                    labels: vec![(
                        *name_span,
                        format!("definition of `{name}`").into(),
                    )],
                    notes: vec![],
                }
            },
            Error::Redefinition { previous, current } => {
                let message = format!("redefinition of `{}`", &source[*current]);
                ParseError {
                    message: message.clone().into(),
                    labels: vec![
                        (*current, message.into()),
                        (
                            *previous,
                            format!("previous definition of `{}`", &source[*previous]).into(),
                        ),
                    ],
                    notes: vec![],
                }
            },
            Error::RecursiveDeclaration { ident, usage } => ParseError {
                message: format!("declaration of `{}` is recursive", &source[*ident]).into(),
                labels: vec![(*ident, "".into()), (*usage, "uses itself here".into())],
                notes: vec![],
            },
            Error::CyclicDeclaration { ident, ref path } => ParseError {
                message: format!("declaration of `{}` is cyclic", &source[*ident]).into(),
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
            Error::TooManyArguments {
                ref function,
                call_span,
                arg_span,
                max_arguments,
            } => ParseError {
                message: format!("too many arguments passed to `{function}`").into(),
                labels: vec![
                    (*call_span, "".into()),
                    (*arg_span, format!("unexpected argument #{}", max_arguments + 1).into())
                ],
                notes: vec![
                    format!("The `{function}` function accepts at most {max_arguments} argument(s)").into()
                ],
            },
            Error::WrongArgumentType {
                ref function,
                call_span,
                arg_span,
                arg_index,
                ref arg_ty,
                ref allowed,
            } => {
                let message = format!(
                    "wrong type passed as argument #{} to `{function}`",
                    arg_index + 1,
                ).into();
                let labels = vec![
                    (*call_span, "".into()),
                    (*arg_span, format!("argument #{} has type `{arg_ty}`", arg_index + 1).into())
                ];

                let mut notes = vec![];
                notes.push(format!("`{function}` accepts the following types for argument #{}:", arg_index + 1).into());
                notes.extend(allowed.iter().map(|ty| format!("allowed type: {ty}").into()));

                ParseError { message, labels, notes }
            },
            Error::InconsistentArgumentType {
                ref function,
                call_span,
                arg_span,
                arg_index,
                ref arg_ty,
                inconsistent_span,
                inconsistent_index,
                ref inconsistent_ty,
                ref allowed
            } => {
                let message = format!(
                    "inconsistent type passed as argument #{} to `{function}`",
                    arg_index + 1,
                ).into();
                let labels = vec![
                    (*call_span, "".into()),
                    (*arg_span, format!("argument #{} has type {arg_ty}", arg_index + 1).into()),
                    (*inconsistent_span, format!(
                        "this argument has type {inconsistent_ty}, which constrains subsequent arguments"
                    ).into()),
                ];
                let mut notes = vec![
                    format!("Because argument #{} has type {inconsistent_ty}, only the following types", inconsistent_index + 1).into(),
                    format!("(or types that automatically convert to them) are accepted for argument #{}:", arg_index + 1).into(),
                ];
                notes.extend(allowed.iter().map(|ty| format!("allowed type: {ty}").into()));

                ParseError { message, labels, notes }
            }
            Error::FunctionReturnsVoid(span) => ParseError {
                labels: vec![(*span, "".into())],
                notes: vec![
                    "perhaps you meant to call the function in a separate statement?".into(),
                ],
                message: "function does not return any value".into(),
            },
            Error::FunctionMustUseUnused(call) => ParseError {
                labels: vec![(*call, "".into())],
                notes: vec![
                    format!(
                        "function '{}' is declared with `@must_use` attribute",
                        &source[*call],
                    ).into(),
                    "use a phony assignment or declare a value using the function call as the initializer".into(),
                ],
                message: "unused return value from function annotated with @must_use".into(),
            },
            Error::FunctionMustUseReturnsVoid(attr, signature) => ParseError {
                labels: vec![
                    (*attr, "".into()),
                    (*signature, "".into()),
                ],
                notes: vec![
                    "declare a return type or remove the attribute".into(),
                ],
                message: "function annotated with @must_use but does not return any value".into(),
            },
            Error::FunctionMustUseOnNonFunction(attr) => ParseError {
                labels: vec![(*attr, "".into())],
                notes: vec![
                    "place `@must_use` on a function declaration with a return type".into(),
                ],
                message: "attribute `@must_use` is only valid on function declarations".into(),
            },
            Error::InvalidWorkGroupUniformLoad(span) => ParseError {
                labels: vec![(*span, "".into())],
                notes: vec!["passed type must be a workgroup pointer".into()],
                message: "incorrect type passed to workgroupUniformLoad".into(),
            },
            Error::Internal(message) => ParseError {
                notes: vec![(*message).into()],
                message: "internal WGSL front end error".into(),
                labels: vec![],
            },
            Error::AutoConversion(ref error) => {
                // destructuring ensures all fields are handled
                let AutoConversionError {
                    dest_span,
                    ref dest_type,
                    source_span,
                    ref source_type,
                } = **error;
                ParseError {
                    message: format!(
                        "automatic conversions cannot convert `{source_type}` to `{dest_type}`"
                    ).into(),
                    labels: vec![
                        (
                            dest_span,
                            format!("a value of type {dest_type} is required here").into(),
                        ),
                        (
                            source_span,
                            format!("this expression has type {source_type}").into(),
                        ),
                    ],
                    notes: vec![],
                }
            }
            Error::AutoConversionLeafScalar(ref error) => {
                let AutoConversionLeafScalarError {
                    dest_span,
                    ref dest_scalar,
                    source_span,
                    ref source_type,
                } = **error;
                ParseError {
                    message: format!(
                        "automatic conversions cannot convert elements of `{source_type}` to `{dest_scalar}`"
                    ).into(),
                    labels: vec![
                        (
                            dest_span,
                            format!(
                                "a value with elements of type {dest_scalar} is required here"
                            )
                            .into(),
                        ),
                        (
                            source_span,
                            format!("this expression has type {source_type}").into(),
                        ),
                    ],
                    notes: vec![],
                }
            }
            Error::ConcretizationFailed(ref error) => {
                let ConcretizationFailedError {
                    expr_span,
                    ref expr_type,
                    ref concretization_preferences,
                } = **error;
                ParseError {
                    labels: vec![(
                        expr_span,
                        format!("this expression has type {expr_type}").into(),
                    )],
                    notes: concretization_preferences
                        .iter()
                        .map(|&(ref scalar, ref err)|
                            format!("the expression couldn't be converted to have {scalar} scalar type: {err}").into()
                        )
                        .collect(),
                    message: "failed to convert expression to a concrete type".into(),
                }
            }
            Error::ExceededLimitForNestedBraces { span, limit } => ParseError {
                labels: vec![(*span, "limit reached at this brace".into())],
                notes: vec![format!("nesting limit is currently set to {limit}").into()],
                message: "brace nesting limit reached".into(),
            },
            Error::DirectiveAfterFirstGlobalDecl { directive_span } => ParseError {
                labels: vec![(
                    *directive_span,
                    "written after first global declaration".into(),
                )],
                notes: vec![concat!(
                    "global directives are only allowed before global declarations; ",
                    "maybe hoist this closer to the top of the shader module?"
                )
                .into()],
                message: "expected global declaration, but found a global directive".into(),
            },
            Error::EnableExtensionNotYetImplemented { kind, span } => ParseError {
                message: format!(
                    "the `{}` enable-extension is not yet supported",
                    EnableExtension::Unimplemented(*kind).to_ident()
                ).into(),
                labels: vec![(
                    *span,
                    concat!(
                        "this enable-extension specifies standard functionality ",
                        "which is not yet implemented in Naga"
                    )
                    .into(),
                )],
                notes: vec![format!(
                    concat!(
                        "Let Naga maintainers know that you ran into this at ",
                        "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                        "so they can prioritize it!"
                    ),
                    kind.tracking_issue_num()
                ).into()],
            },
            Error::EnableExtensionNotEnabled { kind, span } => ParseError {
                message: format!("the `{}` enable extension is not enabled", kind.to_ident()).into(),
                labels: vec![(
                    *span,
                    format!(
                        concat!(
                            "the `{}` \"Enable Extension\" is needed for this functionality, ",
                            "but it is not currently enabled."
                        ),
                        kind.to_ident()
                    )
                    .into(),
                )],
                notes: if let EnableExtension::Unimplemented(kind) = kind {
                    vec![format!(
                        concat!(
                            "This \"Enable Extension\" is not yet implemented. ",
                            "Let Naga maintainers know that you ran into this at ",
                            "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                            "so they can prioritize it!"
                        ),
                        kind.tracking_issue_num()
                    ).into()]
                } else {
                    vec![
                        format!(
                            "You can enable this extension by adding `enable {};` at the top of the shader, before any other items.",
                            kind.to_ident()
                        ).into(),
                    ]
                },
            },
            Error::LanguageExtensionNotYetImplemented { kind, span } => ParseError {
                message: format!(
                    "the `{}` language extension is not yet supported",
                    LanguageExtension::Unimplemented(*kind).to_ident()
                ).into(),
                labels: vec![(*span, "".into())],
                notes: vec![format!(
                    concat!(
                        "Let Naga maintainers know that you ran into this at ",
                        "<https://github.com/gfx-rs/wgpu/issues/{}>, ",
                        "so they can prioritize it!"
                    ),
                    kind.tracking_issue_num()
                ).into()],
            },
            Error::DiagnosticInvalidSeverity {
                severity_control_name_span,
            } => ParseError {
                labels: vec![(
                    *severity_control_name_span,
                    "not a valid severity level".into(),
                )],
                notes: vec![concat!(
                    "See available severities at ",
                    "<https://www.w3.org/TR/WGSL/#diagnostic-severity>."
                )
                .into()],
                message: "invalid `diagnostic(…)` severity".into(),
            },
            Error::DiagnosticDuplicateTriggeringRule(ConflictingDiagnosticRuleError {
                triggering_rule_spans,
            }) => {
                let [first_span, second_span] = triggering_rule_spans;
                ParseError {
                    labels: vec![
                        (*first_span, "first rule".into()),
                        (*second_span, "second rule".into()),
                    ],
                    notes: vec![
                        concat!(
                            "Multiple `diagnostic(…)` rules with the same rule name ",
                            "conflict unless they are directives and the severity is the same.",
                        )
                        .into(),
                        "You should delete the rule you don't want.".into(),
                    ],
                    message: "found conflicting `diagnostic(…)` rule(s)".into(),
                }
            }
            Error::DiagnosticAttributeNotYetImplementedAtParseSite {
                site_name_plural,
                ref spans,
            } => ParseError {
                labels: {
                    let mut spans = spans.iter().cloned();
                    let first = spans
                        .next()
                        .map(|span| {
                            (
                                span,
                                format!("can't use this on {site_name_plural} (yet)").into(),
                            )
                        })
                        .expect("internal error: diag. attr. rejection on empty map");
                    core::iter::once(first)
                        .chain(spans.map(|span| (span, "".into())))
                        .collect()
                },
                notes: vec![concat!(
                    "Let Naga maintainers know that you ran into this at ",
                    "<https://github.com/gfx-rs/wgpu/issues/5320>, ",
                    "so they can prioritize it!"
                ).into()],
                message: "`@diagnostic(…)` attribute(s) not yet implemented".into(),
            },
            Error::DiagnosticAttributeNotSupported { on_what, ref spans } => {
                // In this case the user may have intended to create a global diagnostic filter directive,
                // so display a note to them suggesting the correct syntax.
                let intended_diagnostic_directive = match on_what {
                    DiagnosticAttributeNotSupportedPosition::SemicolonInModulePosition => true,
                    DiagnosticAttributeNotSupportedPosition::Other { .. } => false,
                };
                let on_what_plural = match on_what {
                    DiagnosticAttributeNotSupportedPosition::SemicolonInModulePosition => {
                        "semicolons"
                    }
                    DiagnosticAttributeNotSupportedPosition::Other { display_plural } => {
                        display_plural
                    }
                };
                ParseError {
                    message: format!(
                        "`@diagnostic(…)` attribute(s) on {on_what_plural} are not supported",
                    ).into(),
                    labels: spans
                        .iter()
                        .cloned()
                        .map(|span| (span, "".into()))
                        .collect(),
                    notes: vec![
                        concat!(
                            "`@diagnostic(…)` attributes are only permitted on `fn`s, ",
                            "some statements, and `switch`/`loop` bodies."
                        )
                        .into(),
                        {
                            if intended_diagnostic_directive {
                                concat!(
                                    "If you meant to declare a diagnostic filter that ",
                                    "applies to the entire module, move this line to ",
                                    "the top of the file and remove the `@` symbol."
                                )
                                .into()
                            } else {
                                concat!(
                                    "These attributes are well-formed, ",
                                    "you likely just need to move them."
                                )
                                .into()
                            }
                        },
                    ],
                }
            }
            Error::SelectUnexpectedArgumentType { arg_span, ref arg_type } => ParseError {
                labels: vec![(*arg_span, format!("this value of type {arg_type}").into())],
                notes: vec!["expected a scalar or a `vecN` of scalars".into()],
                message: "unexpected argument type for `select` call".into(),
            },
            Error::SelectRejectAndAcceptHaveNoCommonType {
                reject_span,
                ref reject_type,
                accept_span,
                ref accept_type,
            } => ParseError {
                labels: vec![
                    (*reject_span, format!("reject value of type {reject_type}").into()),
                    (*accept_span, format!("accept value of type {accept_type}").into()),
                ],
                message: "type mismatch for reject and accept values in `select` call".into(),
                notes: vec![],
            },
            Error::StructMemberTooLarge { member_name_span } => ParseError {
                labels: vec![(*member_name_span, "this member exceeds the maximum size".into())],
                notes: vec![format!(
                    "the maximum size is {} bytes",
                    crate::valid::MAX_TYPE_SIZE
                ).into()],
                message: "struct member is too large".into(),
            },
            Error::TypeTooLarge { span } => ParseError {
                labels: vec![(*span, "this type exceeds the maximum size".into())],
                notes: vec![format!(
                    "the maximum size is {} bytes",
                    crate::valid::MAX_TYPE_SIZE
                ).into()],
                message: "type is too large".into(),
            },
            Error::UnderspecifiedCooperativeMatrix => ParseError {
                labels: vec![],
                notes: vec!["must be F32".into()],
                message: "cooperative matrix constructor is underspecified".into(),
            },
            Error::InvalidCooperativeLoadType(span) => ParseError {
                labels: vec![(*span, "type needs the coop_mat<...>".into())],
                notes: vec!["must be a valid cooperative type".into()],
                message: "cooperative load should have a generic type for coop_mat".into(),
            },
            Error::UnsupportedCooperativeScalar(span) => ParseError {
                labels: vec![(*span, "type needs the scalar type specified".into())],
                notes: vec!["must be F32".into()],
                message: "cooperative scalar type is not supported".into(),
            },
            Error::UnusedArgsForTemplate(ref expr_spans) => ParseError {
                labels: expr_spans.iter().cloned().map(|span| -> (_, _){ (span, "unused".into()) }).collect(),
                message: "unused expressions for template".into(),
                notes: vec![],
            },
            Error::UnterminatedBlockComment(span) => ParseError {
                labels: vec![(
                    *span,
                    "must be closed with `*/`".into(),
                )],
                message: "unterminated block comment".into(),
                notes: vec![],
            },
        }
    }
}
