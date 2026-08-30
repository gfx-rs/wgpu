//! End-to-end GPU tests for the resource-table (bindless) feature.
//!
//! M0 is Vulkan-only, unchecked-only, and sampled/depth-texture-only (see
//! `plans/resource-table.md` / `plans/m0-notes.md`). Every test here is gated on
//! `EXPERIMENTAL_SAMPLING_RESOURCE_TABLE | EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED`
//! (so unsupported adapters auto-skip) and skipped on non-Vulkan backends, via
//! [`common::table_params`]. The negative feature-gating tests use their own
//! parameters (they must run with the feature *off*).

mod binding;
mod common;
mod compute;
mod conflict;
mod features;
mod lifecycle;
mod negative;
mod regression;
mod render;

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    binding::all_tests(tests);
    compute::all_tests(tests);
    conflict::all_tests(tests);
    features::all_tests(tests);
    lifecycle::all_tests(tests);
    negative::all_tests(tests);
    regression::all_tests(tests);
    render::all_tests(tests);
}
