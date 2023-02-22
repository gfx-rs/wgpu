const GL_UNMASKED_VENDOR_WEBGL: u32 = 0x9245;
const GL_UNMASKED_RENDERER_WEBGL: u32 = 0x9246;

extern "C" {
    /// returns 1 if success. 0 if failure. extension name must be null terminated
    fn emscripten_webgl_enable_extension(
        context: std::ffi::c_int,
        extension: *const std::ffi::c_char,
    ) -> std::ffi::c_int;
    fn emscripten_webgl_get_current_context() -> std::ffi::c_int;
}
/// if we have debug extension and if we can enable it -> use unmasked vendor/renderer
/// https://github.com/gfx-rs/wgpu/issues/3245
/// # Safety:
/// opengl context MUST BE current
pub unsafe fn get_vendor_renderer_constants(
    extensions: &std::collections::HashSet<String>,
) -> (u32, u32) {
    if extensions.contains("WEBGL_debug_renderer_info")
        && unsafe {
            emscripten_webgl_enable_extension(
                emscripten_webgl_get_current_context(),
                "WEBGL_debug_renderer_info\0".as_ptr() as _,
            )
        } == 1
    {
        // if we try querying unmasked constants without enabling extension, we crash on emscripten.
        (GL_UNMASKED_VENDOR_WEBGL, GL_UNMASKED_RENDERER_WEBGL)
    } else {
        (glow::VENDOR, glow::RENDERER)
    }
}
