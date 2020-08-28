//! Module processing functionality.

mod interface;
mod typifier;
mod validator;

pub use typifier::{check_constant_type, ResolveError, Typifier};
pub use validator::{ValidationError, Validator};
