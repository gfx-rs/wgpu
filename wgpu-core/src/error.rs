use core::fmt;
use std::{error::Error, sync::Arc};

use thiserror::Error;

pub struct ErrorFormatter<'a> {
    writer: &'a mut dyn fmt::Write,
}

impl<'a> ErrorFormatter<'a> {
    pub fn error(&mut self, err: &dyn Error) {
        writeln!(self.writer, "    {err}").expect("Error formatting error");
    }
}

pub trait PrettyError: Error + Sized {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
    }
}

pub fn format_pretty_any(writer: &mut dyn fmt::Write, error: &(dyn Error + 'static)) {
    let mut fmt = ErrorFormatter { writer };

    if let Some(pretty_err) = error.downcast_ref::<ContextError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }

    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderCommandError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::binding_model::CreateBindGroupError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) =
        error.downcast_ref::<crate::binding_model::CreatePipelineLayoutError>()
    {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ExecutionError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassErrorInner>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassErrorInner>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderBundleError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::TransferError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::PassErrorScope>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::track::ResourceUsageCompatibilityError>()
    {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::QueryError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }

    // default
    fmt.error(error)
}

#[derive(Debug, Error)]
#[error(
    "In {fn_ident}{}{}{}",
    if self.label.is_empty() { "" } else { ", label = '" },
    self.label,
    if self.label.is_empty() { "" } else { "'" }
)]
pub struct ContextError {
    pub fn_ident: &'static str,
    #[source]
    #[cfg(send_sync)]
    pub source: Box<dyn Error + Send + Sync + 'static>,
    #[cfg(not(send_sync))]
    pub source: Box<dyn Error + 'static>,
    pub label: String,
}

impl PrettyError for ContextError {}

/// Don't use this error type with thiserror's #[error(transparent)]
#[derive(Clone)]
pub struct MultiError {
    inner: Vec<Arc<dyn Error + Send + Sync + 'static>>,
}

impl MultiError {
    pub fn new<T: Error + Send + Sync + 'static>(
        iter: impl ExactSizeIterator<Item = T>,
    ) -> Option<Self> {
        if iter.len() == 0 {
            return None;
        }
        Some(Self {
            inner: iter.map(Box::from).map(Arc::from).collect(),
        })
    }

    pub fn errors(&self) -> Box<dyn Iterator<Item = &(dyn Error + Send + Sync + 'static)> + '_> {
        Box::new(self.inner.iter().map(|e| e.as_ref()))
    }
}

impl fmt::Debug for MultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(&self.inner[0], f)
    }
}

impl fmt::Display for MultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(&self.inner[0], f)
    }
}

impl Error for MultiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.inner[0].source()
    }
}
