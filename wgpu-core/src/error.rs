use alloc::string::ToString as _;
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use core::fmt;

use thiserror::Error;

use alloc::format;
use core::error;

use hashbrown::HashMap;
use wgpu_sync::Mutex;
use wgt::error::{Error, ErrorFilter, ErrorSource, ErrorType, UncapturedErrorHandler, WebGpuError};
use wgt::WasmNotSendSync;

use crate::device::Device;

/// Implementation of thread IDs for error scope tracking.
///
/// Supports both std and no_std environments, though
/// the no_std implementation is a stub that does not
/// actually distinguish between threads.
mod thread_id {
    #[cfg(feature = "std")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ThreadId(std::thread::ThreadId);

    #[cfg(feature = "std")]
    impl ThreadId {
        pub fn current() -> Self {
            ThreadId(std::thread::current().id())
        }
    }

    #[cfg(not(feature = "std"))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ThreadId(());

    #[cfg(not(feature = "std"))]
    impl ThreadId {
        pub fn current() -> Self {
            // A simple stub implementation for non-std environments. On
            // no_std but multithreaded platforms, this will work, but
            // make error scope global rather than thread-local.
            ThreadId(())
        }
    }
}

struct ErrorScope {
    pub error: Option<Error>,
    pub filter: ErrorFilter,
}

struct InternalErrorSink {
    scopes: HashMap<thread_id::ThreadId, Vec<ErrorScope>>,
    uncaptured_handler: Option<Arc<dyn UncapturedErrorHandler>>,
}

pub struct ErrorSink(Mutex<InternalErrorSink>);

impl ErrorSink {
    pub fn new() -> ErrorSink {
        // The mutex is unranked as it's shortlived
        ErrorSink(Mutex::new(InternalErrorSink::new()))
    }

    #[cold]
    #[track_caller]
    #[inline(never)]
    fn handle_error_inner(
        &self,
        error_type: ErrorType,
        source: ErrorSource,
        label: Option<&str>,
        fn_ident: &'static str,
    ) {
        let source: ErrorSource = Box::new(ContextError {
            fn_ident,
            source,
            label: label.unwrap_or_default().to_string(),
        });
        let final_error_handling = {
            let mut sink = self.0.lock();
            let error = match error_type {
                ErrorType::Internal => {
                    let description = format_error(&*source);
                    Error::Internal {
                        source,
                        description,
                    }
                }
                ErrorType::OutOfMemory => Error::OutOfMemory { source },
                ErrorType::Validation => {
                    let description = format_error(&*source);
                    Error::Validation {
                        source,
                        description,
                    }
                }
                ErrorType::DeviceLost => return, // will be surfaced via callback
            };
            sink.handle_error_or_return_handler(error)
        };

        if let Some(f) = final_error_handling {
            // If the user has provided their own `uncaptured_handler` callback, invoke it now,
            // having released our lock on `sink_mutex`. See the comments on
            // `handle_error_or_return_handler` for details.
            f();
        }
    }

    #[inline]
    #[track_caller]
    pub fn handle_error(
        &self,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        label: Option<&str>,
        fn_ident: &'static str,
    ) {
        let error_type = source.webgpu_error_type();
        self.handle_error_inner(error_type, Box::new(source), label, fn_ident)
    }

    #[inline]
    #[track_caller]
    pub fn handle_error_nolabel(
        &self,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        fn_ident: &'static str,
    ) {
        let error_type = source.webgpu_error_type();
        self.handle_error_inner(error_type, Box::new(source), None, fn_ident)
    }
}

impl InternalErrorSink {
    fn new() -> InternalErrorSink {
        InternalErrorSink {
            scopes: HashMap::new(),
            uncaptured_handler: None,
        }
    }

    /// Deliver the error to
    ///
    /// * the innermost error scope, if any, or
    /// * the uncaptured error handler, if there is one, or
    /// * [`default_error_handler()`].
    ///
    /// If a closure is returned, the caller should call it immediately after dropping the
    /// [`ErrorSink`] mutex guard. This makes sure that the user callback is not called with
    /// a wgpu mutex held.
    #[track_caller]
    #[must_use]
    fn handle_error_or_return_handler(&mut self, err: Error) -> Option<impl FnOnce()> {
        let filter = match err {
            Error::OutOfMemory { .. } => ErrorFilter::OutOfMemory,
            Error::Validation { .. } => ErrorFilter::Validation,
            Error::Internal { .. } => ErrorFilter::Internal,
        };
        let thread_id = thread_id::ThreadId::current();
        let scopes = self.scopes.entry(thread_id).or_default();
        match scopes.iter_mut().rev().find(|scope| scope.filter == filter) {
            Some(scope) => {
                if scope.error.is_none() {
                    scope.error = Some(err);
                }
                None
            }
            None => {
                if let Some(custom_handler) = &self.uncaptured_handler {
                    let custom_handler = Arc::clone(custom_handler);
                    Some(move || (custom_handler)(err))
                } else {
                    // direct call preserves #[track_caller] where dyn can't
                    default_error_handler(err)
                }
            }
        }
    }
}

impl fmt::Debug for InternalErrorSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErrorSink")
    }
}

#[track_caller]
fn default_error_handler(err: Error) -> ! {
    log::error!("Handling wgpu errors as fatal by default");
    panic!("wgpu error: {err}\n");
}

#[derive(Debug, Error)]
#[error("Error scope stack is empty")]
pub struct EmptyErrorScopeStack;

impl Device {
    pub fn on_uncaptured_error(&self, handler: Arc<dyn UncapturedErrorHandler>) {
        let mut error_sink = self.error_sink.0.lock();
        error_sink.uncaptured_handler = Some(handler);
    }

    /// <https://gpuweb.github.io/gpuweb/#dom-gpudevice-pusherrorscope>
    pub fn push_error_scope(&self, filter: ErrorFilter) {
        let mut error_sink = self.error_sink.0.lock();
        let thread_id = thread_id::ThreadId::current();
        let scopes = error_sink.scopes.entry(thread_id).or_default();
        scopes.push(ErrorScope {
            error: None,
            filter,
        });
    }

    /// <https://gpuweb.github.io/gpuweb/#dom-gpudevice-poperrorscope>
    pub fn pop_error_scope(&self) -> Result<Option<Error>, EmptyErrorScopeStack> {
        // 1. If this is lost:
        if !self.is_valid() {
            // Resolve promise with null.
            return Ok(None);
        }
        let mut error_sink = self.error_sink.0.lock();

        let thread_id = thread_id::ThreadId::current();
        let scopes = error_sink.scopes.entry(thread_id).or_default();
        // 2. this.[[errorScopeStack]].size must be > 0.
        match scopes.pop() {
            // 3. Let scope be the result of popping an item off of this.[[errorScopeStack]].
            // 4. Let error be any one of the items in scope.[[errors]], or null if there are none.
            Some(scope) => Ok(scope.error),
            // otherwise Reject promise with an OperationError.
            None => Err(EmptyErrorScopeStack),
        }
    }
}

impl Device {
    // wgpu manipulates the error sink directly
    // to handle panicking
    pub fn error_sink(&self) -> &ErrorSink {
        &self.error_sink
    }
}

#[inline(never)]
pub fn format_error(err: &(dyn error::Error + 'static)) -> String {
    let mut output = String::new();
    let mut level = 1;

    fn print_tree(output: &mut String, level: &mut usize, e: &(dyn error::Error + 'static)) {
        let mut print = |e: &(dyn error::Error + 'static)| {
            use core::fmt::Write;
            writeln!(output, "{}{}", " ".repeat(*level * 2), e).unwrap();

            if let Some(e) = e.source() {
                *level += 1;
                print_tree(output, level, e);
                *level -= 1;
            }
        };
        if let Some(multi) = e.downcast_ref::<MultiError>() {
            for e in multi.errors() {
                print(e);
            }
        } else {
            print(e);
        }
    }

    print_tree(&mut output, &mut level, err);

    format!("Validation Error\n\nCaused by:\n{output}")
}

impl Device {
    #[inline]
    #[track_caller]
    pub fn handle_error(
        &self,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        label: Option<&str>,
        fn_ident: &'static str,
    ) {
        self.error_sink.handle_error(source, label, fn_ident);
    }

    #[inline]
    #[track_caller]
    pub fn handle_error_nolabel(
        &self,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        fn_ident: &'static str,
    ) {
        self.error_sink.handle_error_nolabel(source, fn_ident);
    }
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
    pub source: ErrorSource,
    pub label: String,
}

/// Don't use this error type with thiserror's #[error(transparent)]
#[derive(Clone)]
pub struct MultiError {
    inner: Vec<Arc<dyn error::Error + Send + Sync + 'static>>,
}

impl MultiError {
    pub fn new<T: error::Error + Send + Sync + 'static>(
        iter: impl ExactSizeIterator<Item = T>,
    ) -> Option<Self> {
        if iter.len() == 0 {
            return None;
        }
        Some(Self {
            inner: iter.map(Box::from).map(Arc::from).collect(),
        })
    }

    pub fn errors(
        &self,
    ) -> Box<dyn Iterator<Item = &(dyn error::Error + Send + Sync + 'static)> + '_> {
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

impl error::Error for MultiError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.inner[0].source()
    }
}

// special implementations for wgpu
impl Device {
    pub fn push_error_scope_with_index(&self, filter: ErrorFilter) -> u32 {
        let index = {
            let mut error_sink = self.error_sink.0.lock();
            let thread_id = thread_id::ThreadId::current();
            let scopes = error_sink.scopes.entry(thread_id).or_default();
            scopes
                .len()
                .try_into()
                .expect("Greater than 2^32 nested error scopes")
        };
        self.push_error_scope(filter);
        index
    }

    pub fn pop_error_scope_checked(&self, index: u32) -> Option<Error> {
        #[cfg(feature = "std")]
        fn is_panicking() -> bool {
            std::thread::panicking()
        }

        #[cfg(not(feature = "std"))]
        fn is_panicking() -> bool {
            false
        }

        let mut error_sink = self.error_sink.0.lock();

        // We go out of our way to avoid panicking while unwinding, because that would abort the process,
        // and we are supposed to just drop the error scope on the floor.
        let is_panicking = is_panicking();
        let thread_id = thread_id::ThreadId::current();
        let err = "Mismatched pop_error_scope call: no error scope for this thread. Error scopes are thread-local.";
        let scopes = match error_sink.scopes.get_mut(&thread_id) {
            Some(s) => s,
            None => {
                if !is_panicking {
                    panic!("{err}");
                } else {
                    return None;
                }
            }
        };
        if scopes.is_empty() && !is_panicking {
            panic!("{err}");
        }
        if index as usize != scopes.len() - 1 && !is_panicking {
            panic!(
                "Mismatched pop_error_scope call: error scopes must be popped in reverse order."
            );
        }

        // It would be more correct in this case to use `remove` here so that when unwinding is occurring
        // we would remove the correct error scope, but we don't have such a primitive on the web
        // and having consistent behavior here is more important. If you are unwinding and it unwinds
        // the guards in the wrong order, it's totally reasonable to have incorrect behavior.
        let scope = match scopes.pop() {
            Some(s) => s,
            None if !is_panicking => unreachable!(),
            None => return None,
        };

        scope.error
    }
}
