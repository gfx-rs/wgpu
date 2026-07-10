//! CLI error type and human-readable rendering.

use std::error::Error;
use std::fmt;

/// A simple static-message CLI error.
#[derive(Debug, Clone)]
pub struct CliError(pub &'static str);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for CliError {}

/// Print an error and its source chain to stderr.
#[cold]
#[inline(never)]
pub fn print_err(error: &dyn Error) {
    eprint!("{error}");

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{source}");
        e = source.source();
    }
}
