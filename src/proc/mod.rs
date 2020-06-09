mod interface;
mod typifier;

pub use typifier::{check_constant_types, ResolveError, Typifier, UnexpectedConstantTypeError};
