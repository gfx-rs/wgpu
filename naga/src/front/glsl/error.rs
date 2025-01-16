use super::token::TokenValue;
use crate::SourceLocation;
use crate::{proc::ConstantEvaluatorError, Span};
use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use pp_rs::token::PreprocessorError;
use std::borrow::Cow;
use termcolor::{NoColor, WriteColor};
use thiserror::Error;

fn join_with_comma(list: &[ExpectedToken]) -> String {
    let mut string = "".to_string();
    for (i, val) in list.iter().enumerate() {
        string.push_str(&val.to_string());
        match i {
            i if i == list.len() - 1 => {}
            i if i == list.len() - 2 => string.push_str(" or "),
            _ => string.push_str(", "),
        }
    }
    string
}

/// One of the expected tokens returned in [`InvalidToken`](ErrorKind::InvalidToken).
#[derive(Clone, Debug, PartialEq)]
pub enum ExpectedToken {
    /// A specific token was expected.
    Token(TokenValue),
    /// A type was expected.
    TypeName,
    /// An identifier was expected.
    Identifier,
    /// An integer literal was expected.
    IntLiteral,
    /// A float literal was expected.
    FloatLiteral,
    /// A boolean literal was expected.
    BoolLiteral,
    /// The end of file was expected.
    Eof,
}
impl From<TokenValue> for ExpectedToken {
    fn from(token: TokenValue) -> Self {
        ExpectedToken::Token(token)
    }
}
impl std::fmt::Display for ExpectedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ExpectedToken::Token(ref token) => write!(f, "{token:?}"),
            ExpectedToken::TypeName => write!(f, "a type"),
            ExpectedToken::Identifier => write!(f, "identifier"),
            ExpectedToken::IntLiteral => write!(f, "integer literal"),
            ExpectedToken::FloatLiteral => write!(f, "float literal"),
            ExpectedToken::BoolLiteral => write!(f, "bool literal"),
            ExpectedToken::Eof => write!(f, "end of file"),
        }
    }
}

/// Information about the cause of an error.
#[derive(Clone, Debug, Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ErrorKind {
    /// Whilst parsing as encountered an unexpected EOF.
    #[error("Unexpected end of file")]
    EndOfFile,
    /// The shader specified an unsupported or invalid profile.
    #[error("Invalid profile: {0}")]
    InvalidProfile(String),
    /// The shader requested an unsupported or invalid version.
    #[error("Invalid version: {0}")]
    InvalidVersion(u64),
    /// Whilst parsing an unexpected token was encountered.
    ///
    /// A list of expected tokens is also returned.
    #[error("Expected {expected_tokens}, found {found_token:?}", found_token = .0, expected_tokens = join_with_comma(.1))]
    InvalidToken(TokenValue, Vec<ExpectedToken>),
    /// A specific feature is not yet implemented.
    ///
    /// To help prioritize work please open an issue in the github issue tracker
    /// if none exist already or react to the already existing one.
    #[error("Not implemented: {0}")]
    NotImplemented(&'static str),
    /// A reference to a variable that wasn't declared was used.
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),
    /// A reference to a type that wasn't declared was used.
    #[error("Unknown type: {0}")]
    UnknownType(String),
    /// A reference to a non existent member of a type was made.
    #[error("Unknown field: {0}")]
    UnknownField(String),
    /// An unknown layout qualifier was used.
    ///
    /// If the qualifier does exist please open an issue in the github issue tracker
    /// if none exist already or react to the already existing one to help
    /// prioritize work.
    #[error("Unknown layout qualifier: {0}")]
    UnknownLayoutQualifier(String),
    /// Unsupported matrix of the form matCx2
    ///
    /// Our IR expects matrices of the form matCx2 to have a stride of 8 however
    /// matrices in the std140 layout have a stride of at least 16
    #[error("unsupported matrix of the form matCx2 in std140 block layout")]
    UnsupportedMatrixTypeInStd140,
    /// A variable with the same name already exists in the current scope.
    #[error("Variable already declared: {0}")]
    VariableAlreadyDeclared(String),
    /// A semantic error was detected in the shader.
    #[error("{0}")]
    SemanticError(Cow<'static, str>),
    /// An error was returned by the preprocessor.
    #[error("{0:?}")]
    PreprocessorError(PreprocessorError),
    /// The parser entered an illegal state and exited
    ///
    /// This obviously is a bug and as such should be reported in the github issue tracker
    #[error("Internal error: {0}")]
    InternalError(&'static str),
}

impl From<ConstantEvaluatorError> for ErrorKind {
    fn from(err: ConstantEvaluatorError) -> Self {
        ErrorKind::SemanticError(err.to_string().into())
    }
}

/// Error returned during shader parsing.
#[derive(Clone, Debug, Error)]
#[error("{kind}")]
#[cfg_attr(test, derive(PartialEq))]
pub struct Error {
    /// Holds the information about the error itself.
    pub kind: ErrorKind,
    /// Holds information about the range of the source code where the error happened.
    pub meta: Span,
}

impl Error {
    /// Returns a [`SourceLocation`] for the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        Some(self.meta.location(source))
    }
}

/// A collection of errors returned during shader parsing.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct ParseErrors {
    pub errors: Vec<Error>,
}

impl ParseErrors {
    pub fn emit_to_writer(&self, writer: &mut impl WriteColor, source: &str) {
        self.emit_to_writer_with_path(writer, source, "glsl");
    }

    pub fn emit_to_writer_with_path(&self, writer: &mut impl WriteColor, source: &str, path: &str) {
        let path = path.to_string();
        let files = SimpleFile::new(path, source);
        let config = term::Config::default();

        for err in &self.errors {
            let mut diagnostic = Diagnostic::error().with_message(err.kind.to_string());

            if let Some(range) = err.meta.to_range() {
                diagnostic = diagnostic.with_labels(vec![Label::primary((), range)]);
            }

            term::emit(writer, &config, &files, &diagnostic).expect("cannot write error");
        }
    }

    pub fn emit_to_string(&self, source: &str) -> String {
        let mut writer = NoColor::new(Vec::new());
        self.emit_to_writer(&mut writer, source);
        String::from_utf8(writer.into_inner()).unwrap()
    }
}

impl std::fmt::Display for ParseErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.errors.iter().try_for_each(|e| write!(f, "{e:?}"))
    }
}

impl std::error::Error for ParseErrors {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<Vec<Error>> for ParseErrors {
    fn from(errors: Vec<Error>) -> Self {
        Self { errors }
    }
}
