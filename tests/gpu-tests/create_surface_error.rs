//! Test that `create_surface_*()` accurately reports those errors we can provoke.

/// This test applies to those cfgs that can create a surface from a canvas, which
/// include WebGL and WebGPU, but *not* Emscripten GLES.
#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
#[wasm_bindgen_test::wasm_bindgen_test]
fn canvas_get_context_returned_null() {
    // Not using the normal testing infrastructure because that goes straight to creating the canvas for us.
    let instance = wgpu_test::initialize_instance(wgpu::Backends::all(), false);
    // Create canvas
    let canvas = wgpu_test::initialize_html_canvas();

    // Using a context id that is not "webgl2" or "webgpu" will render the canvas unusable by wgpu.
    canvas.get_context("2d").unwrap();

    #[allow(clippy::redundant_clone)] // false positive — can't and shouldn't move out.
    let error = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("canvas.getContext() returned null"),
        "{error}"
    );
}
