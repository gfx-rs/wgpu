//! End-to-end GPU tests for the resource-table (bindless) feature.
//!
//! M0 is Vulkan-only, unchecked-only, and sampled-texture-only (see
//! `plans/resource-table.md` / `plans/m0-notes.md`). Every test here is gated on
//! `EXPERIMENTAL_SAMPLING_RESOURCE_TABLE | EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED`
//! (so unsupported adapters auto-skip) and skipped on non-Vulkan backends.

mod compute;

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    compute::all_tests(tests);
}
