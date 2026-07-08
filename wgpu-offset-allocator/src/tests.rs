//! In-crate unit and property tests. See submodules.

// The property tests depend on `proptest`, which (via `wait-timeout`) does not build
// on `wasm32`. Gate the module so the crate still tests on wasm; the rest of the suite
// runs everywhere.
#[cfg(not(target_arch = "wasm32"))]
mod proptests;
mod regressions;
mod tlsf_tests;
mod virtual_tests;
