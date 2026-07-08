//! In-crate unit and property tests. See submodules.

mod mock;
#[cfg(not(target_arch = "wasm32"))]
mod proptests;
mod unit;
