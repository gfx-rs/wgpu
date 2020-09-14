//! Module processing functionality.

mod interface;
mod typifier;
mod validator;

pub use interface::{Interface, Visitor};
pub use typifier::{check_constant_type, ResolveContext, ResolveError, Typifier};
pub use validator::{ValidationError, Validator};
