use alloc::{boxed::Box, string::String};
use core::{error::Error, fmt};

#[derive(Clone, Debug)]
pub struct ShaderError<E> {
    /// The source code of the shader.
    pub source: String,
    pub label: Option<String>,
    pub inner: Box<E>,
}

#[cfg(feature = "wgsl-in")]
impl fmt::Display for ShaderError<crate::front::wgsl::ParseError> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}
#[cfg(feature = "glsl-in")]
impl fmt::Display for ShaderError<crate::front::glsl::ParseErrors> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}
#[cfg(feature = "spv-in")]
impl fmt::Display for ShaderError<crate::front::spv::Error> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}
impl fmt::Display for ShaderError<crate::WithSpan<crate::valid::ValidationError>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use codespan_reporting::{files::SimpleFile, term};

        let label = self.label.as_deref().unwrap_or_default();
        let files = SimpleFile::new(label, &self.source);
        let config = term::Config::default();

        let writer = {
            let mut writer = DiagnosticBuffer::new();
            term::emit(
                writer.inner_mut(),
                &config,
                &files,
                &self.inner.diagnostic(),
            )
            .expect("cannot write error");
            writer.into_string()
        };

        write!(f, "\nShader validation {}", writer)
    }
}

#[cfg(feature = "termcolor")]
type DiagnosticBufferInner = codespan_reporting::term::termcolor::NoColor<alloc::vec::Vec<u8>>;
#[cfg(all(not(feature = "termcolor"), feature = "stderr"))]
type DiagnosticBufferInner = alloc::vec::Vec<u8>;
#[cfg(not(any(feature = "termcolor", feature = "stderr")))]
type DiagnosticBufferInner = String;

pub(crate) struct DiagnosticBuffer {
    inner: DiagnosticBufferInner,
}

impl DiagnosticBuffer {
    #[cfg_attr(
        not(feature = "termcolor"),
        expect(
            clippy::missing_const_for_fn,
            reason = "`NoColor::new` isn't `const`, but other `inner`s are."
        )
    )]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "termcolor")]
            inner: codespan_reporting::term::termcolor::NoColor::new(alloc::vec::Vec::new()),
            #[cfg(all(not(feature = "termcolor"), feature = "stderr"))]
            inner: alloc::vec::Vec::new(),
            #[cfg(not(any(feature = "termcolor", feature = "stderr")))]
            inner: String::new(),
        }
    }

    pub fn inner_mut(&mut self) -> &mut DiagnosticBufferInner {
        &mut self.inner
    }

    pub fn into_string(self) -> String {
        let Self { inner } = self;
        #[cfg(feature = "termcolor")]
        let converted = String::from_utf8(inner.into_inner()).unwrap();
        #[cfg(all(not(feature = "termcolor"), feature = "stderr"))]
        let converted = String::from_utf8(inner).unwrap();
        #[cfg(not(any(feature = "termcolor", feature = "stderr")))]
        let converted = inner;

        converted
    }
}
impl<E> Error for ShaderError<E>
where
    ShaderError<E>: fmt::Display,
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.inner)
    }
}
