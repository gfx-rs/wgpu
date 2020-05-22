mod interface;
mod typifier;

pub use typifier::{ResolveError, Typifier, UnexpectedConstantTypeError, check_constant_types};
