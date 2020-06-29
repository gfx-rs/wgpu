//! Module processing functionality.

mod interface;
mod typifier;
mod validator;

pub use typifier::{check_constant_types, ResolveError, Typifier, UnexpectedConstantTypeError};
pub use validator::{ValidationError, Validator};
