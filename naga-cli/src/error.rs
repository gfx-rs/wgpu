//! CLI error type and human-readable rendering.

use codespan_reporting::term::termcolor::{
    Buffer, Color, ColorChoice, ColorSpec, StandardStream, WriteColor,
};
use std::error::Error;
use std::fmt;
use std::io::{IsTerminal as _, Write as _};
use std::sync::OnceLock;

/// A simple static-message CLI error.
#[derive(Debug, Clone)]
pub struct CliError(pub &'static str);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for CliError {}

static COLOR: OnceLock<bool> = OnceLock::new();

/// Decide once whether stderr output is colored.
pub fn init_color(choice: crate::cli::ColorArg) {
    use crate::cli::ColorArg;
    let enabled = match choice {
        ColorArg::Always => true,
        ColorArg::Never => false,
        ColorArg::Auto => std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
    };
    let _ = COLOR.set(enabled);
}

/// A stderr stream that is colored according to `--color`.
pub fn stderr() -> StandardStream {
    StandardStream::stderr(if color_enabled() {
        ColorChoice::Always
    } else {
        ColorChoice::Never
    })
}

fn color_enabled() -> bool {
    COLOR.get().copied().unwrap_or(false)
}

/// Render a diagnostic into a string, with ANSI colors if stderr output is colored.
pub fn render(emit: impl FnOnce(&mut Buffer)) -> String {
    let mut buffer = if color_enabled() {
        Buffer::ansi()
    } else {
        Buffer::no_color()
    };
    emit(&mut buffer);
    String::from_utf8_lossy(buffer.as_slice()).into_owned()
}

fn write_prefix(w: &mut StandardStream, label: &str, color: Color) {
    let _ = w.set_color(ColorSpec::new().set_fg(Some(color)).set_bold(true));
    let _ = write!(w, "{label}");
    let _ = w.reset();
    let _ = write!(w, ": ");
}

/// Print a `warning: ...` line to stderr.
pub fn warn(message: &str) {
    let mut w = stderr();
    write_prefix(&mut w, "warning", Color::Yellow);
    let _ = writeln!(w, "{message}");
}

/// Print an error and its source chain to stderr.
#[cold]
#[inline(never)]
pub fn print_err(error: &dyn Error) {
    let mut w = stderr();
    write_prefix(&mut w, "error", Color::Red);
    let _ = write!(w, "{error}");

    let mut e = error.source();
    if e.is_some() {
        let _ = writeln!(w, ": ");
    } else {
        let _ = writeln!(w);
    }

    while let Some(source) = e {
        let _ = writeln!(w, "\t{source}");
        e = source.source();
    }
}
