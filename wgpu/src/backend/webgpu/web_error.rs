//! Error routing for errors produced by the Rust WebGPU backend.
//!
//! Browser-produced errors are normally routed by the browser's error scope stack and `GPUDevice.onuncapturederror`.
//! However, the backend also produces errors itself when a wgpu operation cannot be represented by the browser API.
//! [`WebErrorSink`] mirrors public error scopes in Rust so those synthetic errors follow the same observable routing rules.

use alloc::{
    boxed::Box,
    format,
    rc::{Rc, Weak},
    string::{String, ToString as _},
    sync::Arc,
    vec::Vec,
};
use core::{cell::RefCell, fmt};

use wasm_bindgen::{prelude::Closure, JsCast as _};

use crate::{
    backend::webgpu::webgpu_sys::{GpuDevice, GpuUncapturedErrorEvent},
    Error, ErrorFilter, UncapturedErrorHandler,
};

/// Error source for failures detected by the Rust WebGPU backend.
///
/// Unlike browser-generated `GPUError` objects, these errors originate in the
/// descriptor mapping or unsupported-operation paths before the browser can
/// produce an error of its own.
#[derive(Debug)]
pub struct WebBackendError(pub String);

impl fmt::Display for WebBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for WebBackendError {}

/// Builds a validation error with the backend operation that detected it.
pub fn web_validation_error(operation: &'static str, message: impl fmt::Display) -> Error {
    let message = message.to_string();
    let description = format!("In {operation}: {message}");
    Error::Validation {
        source: Box::new(WebBackendError(message)),
        description,
    }
}

/// Builds an internal error with the backend operation that detected it.
pub fn web_internal_error(operation: &'static str, message: impl fmt::Display) -> Error {
    let message = message.to_string();
    let description = format!("In {operation}: {message}");
    Error::Internal {
        source: Box::new(WebBackendError(message)),
        description,
    }
}

/// A mirrored public error scope and the first synthetic error it captured.
struct WebErrorScope {
    filter: ErrorFilter,
    error: Option<Error>,
}

/// Mutable routing state shared by a device and its child objects.
#[derive(Default)]
struct WebErrorState {
    /// Public error scopes, ordered from outermost to innermost.
    scopes: Vec<WebErrorScope>,
    /// The handler most recently registered through `Device::on_uncaptured_error`.
    uncaptured_handler: Option<Arc<dyn UncapturedErrorHandler>>,
}

/// Owns the browser callback installed for the lifetime of the error sink.
///
/// The callback captures a weak reference to the sink, avoiding an ownership
/// cycle. Keeping the device here also ensures that the callback remains
/// installed while child backend objects retain the sink.
struct BrowserUncapturedErrorBridge {
    /// A clone retained so the installed callback and its device have the same lifetime.
    device: GpuDevice,
    /// The Rust allocation backing the JavaScript callback.
    closure: Closure<dyn FnMut(GpuUncapturedErrorEvent)>,
}

impl BrowserUncapturedErrorBridge {
    /// Installs the browser callback, forwarding events without extending the sink's lifetime.
    fn install(
        device: &GpuDevice,
        error_sink: Weak<WebErrorSink>,
        map_error: fn(js_sys::Object) -> Error,
    ) -> Self {
        let closure = Closure::wrap(Box::new(move |event: GpuUncapturedErrorEvent| {
            if let Some(error_sink) = error_sink.upgrade() {
                error_sink.report_browser_uncaptured(map_error(event.error().value_of()));
            }
        }) as Box<dyn FnMut(_)>);

        device.set_onuncapturederror(Some(closure.as_ref().unchecked_ref()));

        Self {
            device: device.clone(),
            closure,
        }
    }
}

impl Drop for BrowserUncapturedErrorBridge {
    fn drop(&mut self) {
        let installed = self.closure.as_ref().unchecked_ref::<js_sys::Function>();
        if self
            .device
            .onuncapturederror()
            .is_some_and(|current| current == *installed)
        {
            self.device.set_onuncapturederror(None);
        }
    }
}

/// Routes synthetic backend errors through mirrored scopes and uncaptured handlers.
///
/// Each `WebDevice` has one sink shared with its queue and any child objects
/// that can originate synthetic errors. Browser errors that have already been
/// classified as uncaptured bypass the mirrored scopes: re-filtering them when
/// the browser event is delivered could capture them in a scope pushed later.
pub struct WebErrorSink {
    /// Scope and callback state mutated on the Wasm event-loop thread.
    state: RefCell<WebErrorState>,
    /// Keeps the one browser callback installed for this sink alive.
    _browser_uncaptured_bridge: BrowserUncapturedErrorBridge,
}

/// Why popping a mirrored error scope could not satisfy the indexed LIFO contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PopErrorScopeError {
    /// No mirrored error scope exists.
    Empty,
    /// The requested scope is not the innermost scope.
    NotTop,
}

impl fmt::Debug for WebErrorSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("WebErrorSink")
            .field("scope_count", &state.scopes.len())
            .field(
                "has_uncaptured_handler",
                &state.uncaptured_handler.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl WebErrorSink {
    /// Creates a shared sink and installs its single browser uncaptured-error bridge.
    pub fn new(device: &GpuDevice, map_browser_error: fn(js_sys::Object) -> Error) -> Rc<Self> {
        Rc::new_cyclic(|error_sink| Self {
            state: RefCell::new(WebErrorState::default()),
            _browser_uncaptured_bridge: BrowserUncapturedErrorBridge::install(
                device,
                error_sink.clone(),
                map_browser_error,
            ),
        })
    }

    /// Delivers a backend-produced error to the innermost matching scope.
    pub fn report_synthetic_error(&self, error: Error) {
        let mut state = self.state.borrow_mut();
        let filter = match &error {
            Error::OutOfMemory { .. } => ErrorFilter::OutOfMemory,
            Error::Validation { .. } => ErrorFilter::Validation,
            Error::Internal { .. } => ErrorFilter::Internal,
        };

        // A scope retains only its first synthetic error.
        if let Some(scope) = state
            .scopes
            .iter_mut()
            .rev()
            .find(|scope| scope.filter == filter)
        {
            if scope.error.is_none() {
                scope.error = Some(error);
            }
            return;
        }

        // Invoking user code while `state` is borrowed would make callbacks that
        // re-enter error handling panic on the next `RefCell` borrow. Clone the
        // handler's `Arc`, explicitly release the state borrow, and only then call it.
        let handler = state.uncaptured_handler.clone();
        drop(state);
        report_uncaptured(error, handler);
    }

    /// Forwards an error the browser has already classified as uncaptured.
    ///
    /// This deliberately bypasses mirrored scopes because the browser made its
    /// capture decision when the error occurred, before queuing this event.
    fn report_browser_uncaptured(&self, error: Error) {
        let handler = self.state.borrow().uncaptured_handler.clone();
        report_uncaptured(error, handler);
    }

    /// Replaces the handler used for subsequently delivered uncaptured errors.
    pub fn set_uncaptured_handler(&self, handler: Arc<dyn UncapturedErrorHandler>) {
        self.state.borrow_mut().uncaptured_handler = Some(handler);
    }

    /// Pushes a mirrored scope and returns its zero-based stack index.
    pub fn push_scope(&self, filter: ErrorFilter) -> u32 {
        let mut state = self.state.borrow_mut();
        let index =
            u32::try_from(state.scopes.len()).expect("Greater than 2^32 nested error scopes");
        state.scopes.push(WebErrorScope {
            filter,
            error: None,
        });
        index
    }

    /// Pops the indexed scope and returns its captured synthetic error.
    ///
    /// The scope is left in place if `index` does not identify the innermost
    /// scope, allowing the caller to report the public guard-contract violation.
    pub fn pop_scope(&self, index: u32) -> Result<Option<Error>, PopErrorScopeError> {
        let mut state = self.state.borrow_mut();
        let top_index = state
            .scopes
            .len()
            .checked_sub(1)
            .ok_or(PopErrorScopeError::Empty)?;
        if index as usize != top_index {
            return Err(PopErrorScopeError::NotTop);
        }
        Ok(state.scopes.pop().and_then(|scope| scope.error))
    }
}

/// Invokes the registered handler without an active state borrow, or preserves
/// wgpu's cross-backend fatal default when no handler has been registered.
fn report_uncaptured(error: Error, handler: Option<Arc<dyn UncapturedErrorHandler>>) {
    if let Some(handler) = handler {
        handler(error);
    } else {
        log::error!("Handling wgpu errors as fatal by default");
        panic!("wgpu error: {error}\n");
    }
}
